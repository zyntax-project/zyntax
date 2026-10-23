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
//!
//! A module may also carry itself already lowered. Compiling a program
//! then links against that HIR instead of lowering the module again,
//! which is most of what compiling a small program costs. The HIR is
//! only read by the compiler that produced it, for the pointer width
//! it was produced for. An incompatible artifact must be rebuilt or
//! replaced by checking its original source.

use crate::compiled_artifact::{CompiledArtifactError, CompiledImport};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use zyntax_compiler::hir::HirModule;

mod cache;
#[cfg(not(target_arch = "wasm32"))]
mod preparation;
use cache::CachedInputs;
pub use cache::{SnapshotCache, SnapshotDependency, SnapshotSource};
#[cfg(not(target_arch = "wasm32"))]
pub use preparation::SnapshotPreparation;

const MAGIC: &[u8; 5] = b"ZSNAP";
const SCHEMA_VERSION: u32 = 14;
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

    #[error(
        "module '{module}' has HIR from compiler {found}; this compiler is {expected}; rebuild the snapshot or compile its original source"
    )]
    IncompatibleBuild {
        module: String,
        found: String,
        expected: &'static str,
    },

    #[error(
        "module '{module}' has HIR for {found}-byte pointers; this target uses {expected}-byte pointers; rebuild the snapshot for this target"
    )]
    IncompatiblePointerSize {
        module: String,
        found: u8,
        expected: usize,
    },

    #[error("failed to encode snapshot: {0}")]
    Encode(String),

    #[error("failed to decode snapshot: {0}")]
    Decode(String),

    #[error("failed to lower module '{module}' for the snapshot: {reason}")]
    Lowering { module: String, reason: String },

    #[error(
        "module '{module}' declares effects or handlers, which a snapshot cannot carry lowered"
    )]
    Effectful { module: String },

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
    /// Absent for a language that parses on its own.
    grammar: Option<Extent>,
    modules: Vec<DirectoryEntry>,
    cache: Option<CachedInputs>,
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
    /// The module without top-level function declarations, when each
    /// function declaration is available separately in `signatures`.
    interface: Option<Extent>,
    /// The module's source, when the language chose to carry it.
    source: Option<Extent>,
    /// The module lowered, when the build did that.
    lowered: Option<LoweredEntry>,
    /// The module's function declarations one by one, when it was
    /// lowered: a [`SignatureIndex`] and what it addresses.
    signatures: Option<Extent>,
}

/// A module's function declarations, each encoded on its own and found
/// by name: postcard of this, then the bytes its extents address.
///
/// Typing a call needs the callee's declaration and no other, so a
/// reader decodes the index and the declarations it asks for.
#[derive(Serialize, Deserialize, Default)]
struct SignatureIndex {
    /// Name, declaration extent, and its position in the original
    /// program, ordered by name and then position.
    entries: Vec<(Extent, Extent, u32)>,
    /// Original positions of the declarations retained in the compact
    /// interface, in their encoded order.
    non_functions: Vec<u32>,
}

/// Encode the function declarations of a program whose bodies were
/// stripped, preserving declarations that share a name.
fn signature_table(program: &zyntax_typed_ast::TypedProgram) -> Result<Vec<u8>, SnapshotError> {
    use zyntax_typed_ast::TypedDeclaration;
    let mut named: Vec<(
        String,
        u32,
        &zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>,
    )> = Vec::new();
    let mut non_functions = Vec::new();
    for (position, decl) in program.declarations.iter().enumerate() {
        if let TypedDeclaration::Function(function) = &decl.node
            && let Some(name) = function.name.resolve_global()
        {
            named.push((name, position as u32, decl));
        } else {
            non_functions.push(position as u32);
        }
    }
    named.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut data = Vec::new();
    let mut index = SignatureIndex {
        entries: Vec::new(),
        non_functions,
    };
    for (name, position, declaration) in named {
        let name_extent = Extent::of(name.as_bytes(), data.len());
        data.extend_from_slice(name.as_bytes());
        let encoded =
            postcard::to_allocvec(declaration).map_err(|e| SnapshotError::Encode(e.to_string()))?;
        let decl_extent = Extent::of(&encoded, data.len());
        data.extend_from_slice(&encoded);
        index.entries.push((name_extent, decl_extent, position));
    }
    let mut bytes =
        postcard::to_allocvec(&index).map_err(|e| SnapshotError::Encode(e.to_string()))?;
    bytes.extend_from_slice(&data);
    Ok(bytes)
}

/// Where a module's HIR sits, and what may read it.
#[derive(Serialize, Deserialize, Clone)]
struct LoweredEntry {
    hir: Extent,
    /// The compiler that produced it; no other reads it.
    build_id: String,
    /// The pointer width it was lowered for, in bytes.
    pointer_size: u8,
}

impl LoweredEntry {
    /// Whether this compiler, targeting what it targets, can use the HIR.
    fn usable_here(&self) -> bool {
        self.build_id == zyntax_compiler::BUILD_ID
            && usize::from(self.pointer_size) == zyntax_compiler::target_pointer_size()
    }

    fn validate(&self, module: &str) -> Result<(), SnapshotError> {
        if self.build_id != zyntax_compiler::BUILD_ID {
            return Err(SnapshotError::IncompatibleBuild {
                module: module.to_owned(),
                found: self.build_id.clone(),
                expected: zyntax_compiler::BUILD_ID,
            });
        }
        let expected = zyntax_compiler::target_pointer_size();
        if usize::from(self.pointer_size) != expected {
            return Err(SnapshotError::IncompatiblePointerSize {
                module: module.to_owned(),
                found: self.pointer_size,
                expected,
            });
        }
        Ok(())
    }
}

impl DirectoryEntry {
    fn validate(&self) -> Result<(), SnapshotError> {
        self.lowered
            .as_ref()
            .map_or(Ok(()), |hir| hir.validate(&self.name))
    }
}

/// A language's grammar and standard library, ready to install.
///
/// Holds the artifact's bytes and reads what it needs out of them. The
/// directory is parsed on load; a grammar or a module is a slice of
/// what is already here, and a module is decoded only when something
/// imports it.
pub struct Snapshot {
    /// The artifact: borrowed when the executable embeds it, owned when
    /// it was read from anywhere else.
    bytes: std::borrow::Cow<'static, [u8]>,
    /// Where the bytes the extents address start: everything after the
    /// directory.
    blobs_at: usize,
    language: String,
    grammar: Option<Extent>,
    modules: Vec<DirectoryEntry>,
    cache: Option<CachedInputs>,
    /// Each module, decoded when something first asks for it. A module
    /// nobody imports is never decoded.
    decoded: Vec<std::sync::OnceLock<Result<CompiledImport, String>>>,
    /// Each lowered module's interface, decoded when a named import or
    /// frontend asks for it.
    interfaces: Vec<std::sync::OnceLock<Result<CompiledImport, String>>>,
    /// Each lowered module's HIR directory, decoded independently of
    /// its typed declarations.
    decoded_hir: Vec<
        std::sync::OnceLock<
            Result<Option<std::sync::Arc<zyntax_compiler::bytecode::LazyModule>>, String>,
        >,
    >,
    /// Each module's signature index, and where the bytes it addresses
    /// start in its table, decoded when first asked for.
    signature_indexes: Vec<std::sync::OnceLock<Result<(SignatureIndex, usize), String>>>,
}

impl Snapshot {
    /// Read the snapshot this build embedded (see [`include_snapshot!`]),
    /// checking it is one and that this build understands it.
    ///
    /// Reads the directory and leaves the rest where it lies: a part is
    /// read in place when something asks for it. The lowered modules are
    /// not checked against their checksums, since the build that wrote
    /// them is the one reading them.
    pub fn load(bytes: &'static [u8]) -> Result<Self, SnapshotError> {
        Self::read(std::borrow::Cow::Borrowed(bytes))
    }

    /// Read a snapshot that arrived from outside the executable, such as
    /// a file. It is kept as it arrived, and a lowered module is checked
    /// against its checksum before it is read.
    pub fn load_owned(bytes: Vec<u8>) -> Result<Self, SnapshotError> {
        Self::read(std::borrow::Cow::Owned(bytes))
    }

    fn read(bytes: std::borrow::Cow<'static, [u8]>) -> Result<Self, SnapshotError> {
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

        for module in &directory.modules {
            module.validate()?;
        }

        let decoded = directory
            .modules
            .iter()
            .map(|_| std::sync::OnceLock::new())
            .collect();
        let interfaces = directory
            .modules
            .iter()
            .map(|_| std::sync::OnceLock::new())
            .collect();
        let decoded_hir = directory
            .modules
            .iter()
            .map(|_| std::sync::OnceLock::new())
            .collect();
        let signature_indexes = directory
            .modules
            .iter()
            .map(|_| std::sync::OnceLock::new())
            .collect();
        Ok(Self {
            bytes,
            blobs_at: directory_end,
            language: directory.language,
            grammar: directory.grammar,
            modules: directory.modules,
            cache: directory.cache,
            decoded,
            interfaces,
            decoded_hir,
            signature_indexes,
        })
    }

    /// Everything after the directory, addressed by the extents in it.
    fn blobs(&self) -> &[u8] {
        &self.bytes[self.blobs_at..]
    }

    /// A module's lowered form, read in place from an embedded snapshot
    /// and from a copy of its bytes otherwise.
    fn lazy_hir(
        &self,
        extent: Extent,
        name: &str,
    ) -> Result<(zyntax_compiler::bytecode::LazyModule, usize), SnapshotError> {
        let decode =
            |e: zyntax_compiler::bytecode::BytecodeError| SnapshotError::Decode(e.to_string());
        match &self.bytes {
            std::borrow::Cow::Borrowed(all) => {
                let blobs: &'static [u8] = &all[self.blobs_at..];
                let bytes = extent.slice(blobs, name)?;
                zyntax_compiler::bytecode::deserialize_module_lazy_trusted(bytes)
                    .map(|hir| (hir, bytes.len()))
                    .map_err(decode)
            }
            std::borrow::Cow::Owned(_) => {
                let bytes = extent.slice(self.blobs(), name)?;
                zyntax_compiler::bytecode::deserialize_module_lazy(bytes.to_vec())
                    .map(|hir| (hir, bytes.len()))
                    .map_err(decode)
            }
        }
    }

    /// The language this snapshot installs.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// The grammar, as [`crate::LanguageGrammar::from_compiled_bytes`]
    /// reads it, when the language ships one.
    pub fn grammar_bytes(&self) -> Option<&[u8]> {
        self.grammar
            .as_ref()
            .and_then(|extent| extent.slice(self.blobs(), "the grammar").ok())
    }

    /// The modules it carries, in the order they were built.
    pub fn module_names(&self) -> impl Iterator<Item = &str> {
        self.modules.iter().map(|m| m.name.as_str())
    }

    /// Checked inputs recorded by a persistent source-project cache.
    pub fn cached_sources(&self) -> Option<&[SnapshotSource]> {
        self.cache.as_ref().map(|cache| cache.sources.as_slice())
    }

    /// Check lowered modules against this compiler and its current target.
    /// Reads only directory metadata, leaving typed declarations and HIR encoded.
    pub fn validate_compatibility(&self) -> Result<(), SnapshotError> {
        for module in &self.modules {
            module.validate()?;
        }
        Ok(())
    }

    /// One module, decoded on first use and kept after. Incompatible lowered
    /// artifacts are rejected; their typed declarations have no executable bodies.
    pub fn module(&self, name: &str) -> Result<Option<CompiledImport>, SnapshotError> {
        self.module_selecting(name, None)
    }

    /// A lowered module's types and non-function declarations. Function
    /// declarations are available individually through [`Self::function_signature`].
    /// Falls back to the whole module for snapshots without an interface.
    pub fn module_interface(&self, name: &str) -> Result<Option<CompiledImport>, SnapshotError> {
        let Some(index) = self.modules.iter().position(|m| m.name == name) else {
            return Ok(None);
        };
        if self.modules[index].interface.is_none() {
            return self.module(name);
        }
        self.interface_at(index).map(Some)
    }

    fn interface_at(&self, index: usize) -> Result<CompiledImport, SnapshotError> {
        self.modules[index].validate()?;
        self.interfaces[index]
            .get_or_init(|| {
                let entry = &self.modules[index];
                let extent = entry.interface.ok_or_else(|| {
                    format!("module '{}' carries no compact interface", entry.name)
                })?;
                let bytes = extent
                    .slice(self.blobs(), &entry.name)
                    .map_err(|e| e.to_string())?;
                let t0 = web_time::Instant::now();
                let interface = CompiledImport::decode(bytes).map_err(|e| e.to_string())?;
                if std::env::var_os("ZYNTAX_TRACE_LOWER_PHASES").is_some() {
                    eprintln!(
                        "[SNAPSHOT] {}: interface {:.2} ms ({} bytes)",
                        entry.name,
                        t0.elapsed().as_secs_f64() * 1000.0,
                        bytes.len(),
                    );
                }
                Ok(interface)
            })
            .as_ref()
            .cloned()
            .map_err(|e| SnapshotError::Decode(e.clone()))
    }

    fn hir_at(
        &self,
        index: usize,
    ) -> Result<Option<Arc<zyntax_compiler::bytecode::LazyModule>>, SnapshotError> {
        self.modules[index].validate()?;
        self.decoded_hir[index]
            .get_or_init(|| {
                let entry = &self.modules[index];
                let Some(lowered) = entry.lowered.as_ref() else {
                    return Ok(None);
                };
                let t0 = web_time::Instant::now();
                self.lazy_hir(lowered.hir, &entry.name)
                    .map(|(hir, bytes)| {
                        if std::env::var_os("ZYNTAX_TRACE_LOWER_PHASES").is_some() {
                            eprintln!(
                                "[SNAPSHOT] {}: hir {:.2} ms ({} bytes)",
                                entry.name,
                                t0.elapsed().as_secs_f64() * 1000.0,
                                bytes,
                            );
                        }
                        Some(Arc::new(hir))
                    })
                    .map_err(|e| e.to_string())
            })
            .as_ref()
            .cloned()
            .map_err(|e| SnapshotError::Decode(e.clone()))
    }

    /// [`Self::module`], keeping of its functions only those `named`
    /// names when that is given and the module arrives lowered.
    pub(crate) fn module_selecting(
        &self,
        name: &str,
        named: Option<&std::collections::HashSet<zyntax_typed_ast::InternedString>>,
    ) -> Result<Option<CompiledImport>, SnapshotError> {
        let Some(index) = self.modules.iter().position(|m| m.name == name) else {
            return Ok(None);
        };
        self.modules[index].validate()?;
        if let Some(named) = named
            && self.modules[index].interface.is_some()
            && let Some(hir) = self.hir_at(index)?
        {
            let mut program = self.interface_at(index)?.into_program();
            let positions = self.non_function_positions_at(index)?;
            if positions.len() != program.declarations.len() {
                return Err(SnapshotError::Decode(format!(
                    "module '{}' has {} compact declarations but {} declaration positions",
                    name,
                    program.declarations.len(),
                    positions.len(),
                )));
            }
            let mut declarations: Vec<_> = positions
                .into_iter()
                .zip(std::mem::take(&mut program.declarations))
                .collect();
            let mut names: Vec<String> = named.iter().filter_map(|n| n.resolve_global()).collect();
            names.sort();
            for name in names {
                declarations.extend(self.declaration_signatures_at(index, &name)?);
            }
            declarations.sort_by_key(|(position, _)| *position);
            program.declarations = declarations
                .into_iter()
                .map(|(_, declaration)| declaration)
                .collect();
            return Ok(Some(
                CompiledImport::new(self.language.clone(), name.to_string(), program).with_hir(hir),
            ));
        }
        self.decoded[index]
            .get_or_init(|| self.decode_at(index).map_err(|e| e.to_string()))
            .as_ref()
            .map(|module| {
                // A module lowered here is linked, so what it does not
                // name is not needed to lower it; one parsed only is.
                Some(match named {
                    Some(named) if module.hir().is_some() => module.selecting(named),
                    _ => module.clone(),
                })
            })
            .map_err(|e| SnapshotError::Decode(e.clone()))
    }

    fn decode_at(&self, index: usize) -> Result<CompiledImport, SnapshotError> {
        let entry = &self.modules[index];
        entry.validate()?;
        let bytes = entry.artifact.slice(self.blobs(), &entry.name)?;
        let trace = std::env::var_os("ZYNTAX_TRACE_LOWER_PHASES").is_some();
        let t0 = web_time::Instant::now();
        let import = CompiledImport::decode(bytes)?;
        let elapsed = t0.elapsed();
        let hir = self.hir_at(index)?;
        if trace {
            eprintln!(
                "[SNAPSHOT] {}: program {:.2} ms ({} bytes)",
                entry.name,
                elapsed.as_secs_f64() * 1000.0,
                bytes.len(),
            );
        }
        Ok(match hir {
            Some(hir) => import.with_hir(hir),
            None => import,
        })
    }

    /// Function `name` as module `module` declares it, without a body:
    /// what typing a call to it needs. Decodes that declaration alone.
    /// `None` when the module declares no such function or the snapshot
    /// carries no declarations for it one by one.
    pub fn function_signature(
        &self,
        module: &str,
        name: &str,
    ) -> Result<Option<zyntax_typed_ast::TypedFunction>, SnapshotError> {
        let Some(at) = self.modules.iter().position(|m| m.name == module) else {
            return Ok(None);
        };
        self.declaration_signatures_at(at, name)
            .map(|declarations| {
                declarations
                    .into_iter()
                    .find_map(|(_, declaration)| match declaration.node {
                        zyntax_typed_ast::TypedDeclaration::Function(function) => Some(function),
                        _ => None,
                    })
            })
    }

    fn signature_index_at(
        &self,
        at: usize,
    ) -> Result<Option<(&SignatureIndex, usize, &[u8])>, SnapshotError> {
        let entry = &self.modules[at];
        entry.validate()?;
        let Some(extent) = entry.signatures else {
            return Ok(None);
        };
        let table = extent.slice(self.blobs(), &entry.name)?;
        let (index, data_at) = self.signature_indexes[at]
            .get_or_init(|| {
                postcard::take_from_bytes::<SignatureIndex>(table)
                    .map(|(index, rest)| (index, table.len() - rest.len()))
                    .map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| SnapshotError::Decode(e.clone()))?;
        Ok(Some((index, *data_at, table)))
    }

    fn non_function_positions_at(&self, at: usize) -> Result<Vec<u32>, SnapshotError> {
        Ok(self
            .signature_index_at(at)?
            .map(|(index, _, _)| index.non_functions.clone())
            .unwrap_or_default())
    }

    fn declaration_signatures_at(
        &self,
        at: usize,
        name: &str,
    ) -> Result<
        Vec<(
            u32,
            zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>,
        )>,
        SnapshotError,
    > {
        let entry = &self.modules[at];
        let Some((index, data_at, table)) = self.signature_index_at(at)? else {
            return Ok(Vec::new());
        };
        let data = &table[data_at..];
        let name_of = |extent: &Extent| extent.slice(data, &entry.name).unwrap_or_default();
        let first = index
            .entries
            .partition_point(|(entry_name, _, _)| name_of(entry_name) < name.as_bytes());
        let mut declarations = Vec::new();
        for (entry_name, declaration, position) in &index.entries[first..] {
            if name_of(entry_name) != name.as_bytes() {
                break;
            }
            let bytes = declaration.slice(data, &entry.name)?;
            declarations.push((
                *position,
                postcard::from_bytes(bytes).map_err(|e| SnapshotError::Decode(e.to_string()))?,
            ));
        }
        Ok(declarations)
    }

    /// Whether a module's lowered form is carried and readable here.
    pub fn carries_hir(&self, name: &str) -> bool {
        self.modules
            .iter()
            .find(|m| m.name == name)
            .and_then(|m| m.lowered.as_ref())
            .is_some_and(LoweredEntry::usable_here)
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
        let bytes = extent.slice(self.blobs(), &entry.name).ok()?;
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

    /// Consume the snapshot, moving decoded modules rather than cloning their bodies.
    pub fn into_modules(mut self) -> Result<Vec<CompiledImport>, SnapshotError> {
        self.validate_compatibility()?;
        if self.modules.len() == 1
            && matches!(self.bytes, std::borrow::Cow::Owned(_))
            && self.decoded[0].get().is_none()
            && self.decoded_hir[0].get().is_none()
            && let Some(lowered) = &self.modules[0].lowered
        {
            let entry = &self.modules[0];
            let import = CompiledImport::decode(entry.artifact.slice(self.blobs(), &entry.name)?)?;
            lowered.hir.slice(self.blobs(), &entry.name)?;
            let start = self.blobs_at + lowered.hir.at as usize;
            let length = lowered.hir.len as usize;
            let std::borrow::Cow::Owned(mut bytes) = std::mem::take(&mut self.bytes) else {
                unreachable!()
            };
            // Reuse the project's owned image for its split HIR, without
            // allocating another copy of all the encoded function bodies.
            bytes.copy_within(start..start + length, 0);
            bytes.truncate(length);
            let hir = zyntax_compiler::bytecode::deserialize_module_lazy(bytes)
                .map_err(|error| SnapshotError::Decode(error.to_string()))?;
            return Ok(vec![import.with_hir(Arc::new(hir))]);
        }
        let mut modules = Vec::with_capacity(self.modules.len());
        for index in 0..self.modules.len() {
            modules.push(match self.decoded[index].take() {
                Some(result) => result.map_err(SnapshotError::Decode)?,
                None => self.decode_at(index)?,
            });
        }
        Ok(modules)
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
    cache: Option<CachedInputs>,
}

/// A module waiting to be written into a snapshot.
struct PendingModule {
    name: String,
    max_type_id: u32,
    artifact: Vec<u8>,
    /// The encoded module without top-level function declarations.
    interface: Option<Vec<u8>>,
    source: Option<String>,
    /// The module's HIR, encoded, with what may read it.
    lowered: Option<(Vec<u8>, String, u8)>,
    /// The module's function declarations, from [`signature_table`].
    signatures: Option<Vec<u8>>,
}

/// Lower a module the way a runtime would, for a snapshot to carry.
///
/// `builtins` are the language's extern aliases (a grammar's
/// `builtins().functions`); `prelowered` are the snapshot's earlier
/// modules that this one imports, already lowered. The result is
/// optimised, so what inlines into a program later is what the runtime
/// would have inlined. A module declaring effects or handlers is
/// refused: their ids are also baked into constants the loader cannot
/// relocate.
///
/// Generic bodies remain unoptimized templates for importing callers to
/// specialize. Unlike whole-program monomorphization, library packaging
/// retains those templates even when no local call instantiates them.
pub fn lower_for_snapshot(
    name: &str,
    program: zyntax_typed_ast::TypedProgram,
    builtins: indexmap::IndexMap<String, String>,
    prelowered: Vec<Arc<zyntax_compiler::bytecode::LazyModule>>,
) -> Result<HirModule, SnapshotError> {
    lower_for_snapshot_releasing(name, program, builtins, prelowered, false, None)
}

/// [`lower_for_snapshot`] with the module's storage released as the
/// runtime that loads it will release: a module compiled with automatic
/// release on is optimised with it on, so the two agree. With
/// `error_flag_global`, the name the runtime gives its error flag, each
/// function records whether its calls leave the flag as they found it,
/// for a program that holds the function as a shell.
pub fn lower_for_snapshot_releasing(
    name: &str,
    mut program: zyntax_typed_ast::TypedProgram,
    builtins: indexmap::IndexMap<String, String>,
    prelowered: Vec<Arc<zyntax_compiler::bytecode::LazyModule>>,
    automatic_release: bool,
    error_flag_global: Option<&str>,
) -> Result<HirModule, SnapshotError> {
    program
        .type_registry
        .set_current_module(Some(zyntax_typed_ast::InternedString::new_global(name)));
    let none_grammars = std::collections::HashMap::new();
    let none_signatures = std::collections::HashMap::new();
    let none_modules = crate::import_chain::SnapshotModules::default();
    let lowered = crate::lower::lower_typed_program(
        program,
        crate::lower::Inputs {
            grammars: &none_grammars,
            plugin_signatures: &none_signatures,
            import_resolvers: &[],
            compiled_import_resolvers: &[],
            host_modules: &[],
            snapshot_modules: &none_modules,
            builtins,
            builtin_registry: Arc::new(
                zyntax_compiler::builtin_class::BuiltinRegistry::with_defaults(),
            ),
            entry_names: Vec::new(),
            closed: false,
            prelowered,
            defer_prelowered_bodies: false,
            keep_generic_templates: true,
            linked: Arc::default(),
            selective: false,
            pattern_rewrites: true,
            error_flag_global: None,
        },
    )
    .map_err(|e| SnapshotError::Lowering {
        module: name.to_string(),
        reason: e.to_string(),
    })?;
    let mut module = lowered.module;
    optimize_for_snapshot(name, &mut module, automatic_release, error_flag_global)?;
    Ok(module)
}

pub(crate) fn optimize_for_snapshot(
    name: &str,
    module: &mut HirModule,
    automatic_release: bool,
    error_flag_global: Option<&str>,
) -> Result<(), SnapshotError> {
    optimize_snapshot_hir(name, module, automatic_release, error_flag_global, None).map(|_| ())
}

/// Optimize an owned HIR candidate. Cancellation leaves it unpublished and discardable.
pub fn optimize_snapshot_hir(
    name: &str,
    module: &mut HirModule,
    automatic_release: bool,
    error_flag_global: Option<&str>,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> Result<bool, SnapshotError> {
    let stopped = || cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire));
    if stopped() {
        return Ok(false);
    }
    if !module.effects.is_empty() || !module.handlers.is_empty() {
        return Err(SnapshotError::Effectful {
            module: name.to_string(),
        });
    }
    module.automatic_release = automatic_release;
    for function in module
        .functions
        .values_mut()
        .filter(|f| !f.attributes.optimized)
    {
        function.attributes.release_facts = None;
    }
    // Opaque template bodies become executable only after type substitution.
    let templates: Vec<_> = module
        .functions
        .extract_if(.., |_, f| {
            !f.signature.type_params.is_empty() || !f.signature.const_params.is_empty()
        })
        .collect();
    if let Some(flag) = cancelled {
        if !zyntax_compiler::run_snapshot_opts_cancellable(module, flag) {
            return Ok(false);
        }
    } else {
        zyntax_compiler::run_interp_safe_opts_keeping_readers(module);
    }
    zyntax_compiler::run_native_only_opts(module);
    if stopped() {
        return Ok(false);
    }
    // The bodies as every program adopts them, then the facts of those
    // bodies: a program that links the module reads them rather than
    // computing them over the library again.
    zyntax_compiler::const_boxes::run_library(module, name);
    let facts = zyntax_compiler::drop_insert::facts_of(module);
    zyntax_compiler::drop_insert::record_facts(module, &facts);
    if let Some(flag) = error_flag_global {
        zyntax_compiler::error_flag::record_keeps(module, flag);
    }
    zyntax_compiler::mark_optimized(module);
    module.functions.extend(templates);
    Ok(!stopped())
}

/// Drop every function body from a program whose HIR travels beside it.
/// What remains types a caller; nothing lowers it.
pub(crate) fn strip_bodies(program: &mut zyntax_typed_ast::TypedProgram) {
    use zyntax_typed_ast::TypedDeclaration;
    program.declarations.retain(|d| !matches!(d.node, TypedDeclaration::Initializer(_))
        && !matches!(&d.node, TypedDeclaration::Function(f) if zyntax_compiler::module_init::is_initializer(f)));
    for decl in &mut program.declarations {
        match &mut decl.node {
            TypedDeclaration::Function(function) => function.body = None,
            TypedDeclaration::Impl(imp) => {
                for method in &mut imp.methods {
                    method.body = None;
                }
            }
            _ => {}
        }
    }
}

impl SnapshotBuilder {
    pub fn new(language: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            grammar: None,
            modules: Vec::new(),
            cache: None,
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

    /// Encode a checked module while the caller retains its program for execution.
    pub fn module_ref(
        mut self,
        name: impl Into<String>,
        program: &zyntax_typed_ast::TypedProgram,
    ) -> Result<Self, SnapshotError> {
        let name = name.into();
        let artifact = CompiledImport::encode_program(&self.language, &name, program)?;
        self.modules.push(PendingModule {
            name,
            max_type_id: program.type_registry.max_type_id(),
            artifact,
            interface: None,
            source: None,
            lowered: None,
            signatures: None,
        });
        Ok(self)
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
        let artifact = CompiledImport::encode_program(&self.language, &name, &program)?;
        self.modules.push(PendingModule {
            name,
            max_type_id: program.type_registry.max_type_id(),
            artifact,
            interface: None,
            source,
            lowered: None,
            signatures: None,
        });
        Ok(self)
    }

    /// Add a module together with its lowered form, from
    /// [`lower_for_snapshot`]. The program's function bodies are left
    /// out: the HIR is what runs, the declarations are what a caller
    /// types against.
    pub fn module_lowered(
        self,
        name: impl Into<String>,
        mut program: zyntax_typed_ast::TypedProgram,
        hir: &HirModule,
    ) -> Result<Self, SnapshotError> {
        strip_bodies(&mut program);
        self.module_lowered_ref(name, &program, hir)
    }

    /// Encode lowered HIR while retaining its body-free declarations for another owner.
    pub fn module_lowered_ref(
        mut self,
        name: impl Into<String>,
        program: &zyntax_typed_ast::TypedProgram,
        hir: &HirModule,
    ) -> Result<Self, SnapshotError> {
        use zyntax_typed_ast::TypedDeclaration;
        let name = name.into();
        if !hir.effects.is_empty() || !hir.handlers.is_empty() {
            return Err(SnapshotError::Effectful { module: name });
        }
        if program.declarations.iter().any(|decl| match &decl.node {
            TypedDeclaration::Function(f) => f.body.is_some(),
            TypedDeclaration::Initializer(_) => true,
            TypedDeclaration::Impl(imp) => imp.methods.iter().any(|f| f.body.is_some()),
            _ => false,
        }) {
            return Err(SnapshotError::Encode(
                "lowered metadata still contains typed bodies".into(),
            ));
        }
        let signatures = signature_table(program)?;
        // Bodies encoded one by one, so a program decodes the ones it
        // reaches and no others.
        let encoded = zyntax_compiler::bytecode::serialize_module(
            hir,
            zyntax_compiler::bytecode::Format::Split,
        )
        .map_err(|e| SnapshotError::Encode(e.to_string()))?;
        let pointer_size = u8::try_from(zyntax_compiler::target_pointer_size())
            .map_err(|e| SnapshotError::Encode(e.to_string()))?;
        let interface = zyntax_typed_ast::TypedProgram {
            declarations: program
                .declarations
                .iter()
                .filter(|d| !matches!(d.node, TypedDeclaration::Function(_)))
                .cloned()
                .collect(),
            language: program.language,
            span: program.span,
            source_files: program.source_files.clone(),
            checker_interface: program.checker_interface.clone(),
            type_registry: program.type_registry.clone(),
        };
        self.modules.push(PendingModule {
            max_type_id: program.type_registry.max_type_id(),
            artifact: CompiledImport::encode_program(&self.language, &name, program)?,
            interface: Some(CompiledImport::encode_program(
                &self.language,
                &name,
                &interface,
            )?),
            name,
            source: None,
            lowered: Some((encoded, zyntax_compiler::BUILD_ID.to_string(), pointer_size)),
            signatures: Some(signatures),
        });
        Ok(self)
    }

    /// Encode the snapshot.
    ///
    /// A directory of names and extents, then everything those extents
    /// point at. Reading the directory is the whole cost of loading;
    /// the parts are read where they lie.
    pub fn encode(self) -> Result<Vec<u8>, SnapshotError> {
        let mut blobs: Vec<u8> = Vec::new();
        let mut put = |bytes: &[u8], blobs: &mut Vec<u8>| {
            let extent = Extent::of(bytes, blobs.len());
            blobs.extend_from_slice(bytes);
            extent
        };

        let grammar_extent = self.grammar.as_ref().map(|bytes| put(bytes, &mut blobs));
        let mut entries = Vec::with_capacity(self.modules.len());
        for module in &self.modules {
            let artifact = put(&module.artifact, &mut blobs);
            let interface = module
                .interface
                .as_ref()
                .map(|bytes| put(bytes, &mut blobs));
            let source = module
                .source
                .as_ref()
                .map(|text| put(text.as_bytes(), &mut blobs));
            let lowered = module
                .lowered
                .as_ref()
                .map(|(bytes, build_id, pointer_size)| LoweredEntry {
                    hir: put(bytes, &mut blobs),
                    build_id: build_id.clone(),
                    pointer_size: *pointer_size,
                });
            let signatures = module
                .signatures
                .as_ref()
                .map(|bytes| put(bytes, &mut blobs));
            entries.push(DirectoryEntry {
                name: module.name.clone(),
                max_type_id: module.max_type_id,
                artifact,
                interface,
                source,
                lowered,
                signatures,
            });
        }

        let mut cache = self.cache;
        if let Some(inputs) = &mut cache {
            inputs.blobs_hash = cache::digest(&blobs);
        }
        let directory = Directory {
            language: self.language,
            grammar: grammar_extent,
            modules: entries,
            cache,
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

    #[test]
    fn raw_lowered_entries_reject_unrelocatable_effects() {
        use zyntax_compiler::hir::{HirEffect, HirId};
        use zyntax_typed_ast::{InternedString, TypedProgram};
        let name = InternedString::new_global("effects");
        let mut hir = HirModule::new(name);
        let id = HirId::new();
        hir.effects.insert(
            id,
            HirEffect {
                id,
                name,
                type_params: vec![],
                operations: vec![],
            },
        );
        assert!(matches!(
            SnapshotBuilder::new("test").module_lowered("effects", TypedProgram::default(), &hir),
            Err(SnapshotError::Effectful { .. })
        ));
        assert!(matches!(
            SnapshotBuilder::new("test").module_lowered_ref(
                "effects",
                &TypedProgram::default(),
                &hir
            ),
            Err(SnapshotError::Effectful { .. })
        ));
    }
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

        let snapshot = Snapshot::load_owned(bytes).expect("load");
        assert_eq!(snapshot.language(), "demo");
        assert_eq!(snapshot.grammar_bytes(), Some(&[1u8, 2, 3][..]));
        assert_eq!(snapshot.module_names().collect::<Vec<_>>(), vec!["prelude"]);
    }

    #[test]
    fn borrowed_programs_encode_and_decoded_programs_move() {
        let mut program = empty_program();
        program
            .source_files
            .push(zyntax_typed_ast::source::SourceFile::new(
                "source.ts".into(),
                "source body".into(),
            ));
        let bytes = SnapshotBuilder::new("demo")
            .module_ref("library", &program)
            .unwrap()
            .module_ref("other", &program)
            .unwrap()
            .encode()
            .unwrap();
        assert_eq!(program.source_files[0].content, "source body");
        let snapshot = Snapshot::load_owned(bytes).unwrap();
        // Populate one lazy slot; consuming it must retain the same source allocation.
        snapshot.module("library").unwrap();
        let pointer = snapshot.decoded[0]
            .get()
            .unwrap()
            .as_ref()
            .unwrap()
            .program()
            .source_files[0]
            .content
            .as_ptr();
        let modules = snapshot.into_modules().unwrap();
        assert_eq!(modules.len(), 2);
        assert_eq!(
            modules[0].program().source_files[0].content.as_ptr(),
            pointer
        );
        assert_eq!(modules[1].program().source_files[0].content, "source body");
        assert_eq!(modules[0].module_name(), "library");
        assert_eq!(modules[1].module_name(), "other");
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

        let snapshot = Snapshot::load_owned(bytes).expect("load");
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
    fn a_snapshot_without_a_grammar_ships_none() {
        // A frontend that parses on its own still ships its library.
        let bytes = SnapshotBuilder::new("demo")
            .module("builtins", empty_program())
            .expect("module")
            .encode()
            .expect("encode");
        let snapshot = Snapshot::load_owned(bytes).expect("load");
        assert_eq!(snapshot.grammar_bytes(), None);
        assert_eq!(
            snapshot.module_names().collect::<Vec<_>>(),
            vec!["builtins"]
        );
    }

    #[test]
    fn a_lowered_module_arrives_with_its_hir() {
        let mut arena = zyntax_typed_ast::AstArena::new();
        let hir = HirModule::new(arena.intern_string("lib"));
        let bytes = SnapshotBuilder::new("demo")
            .module_lowered("lib", empty_program(), &hir)
            .expect("module")
            .encode()
            .expect("encode");
        let snapshot = Snapshot::load_owned(bytes).expect("load");
        assert!(snapshot.carries_hir("lib"));
        let module = snapshot.module("lib").expect("decodes").expect("present");
        assert!(module.hir().is_some(), "the HIR came with the module");
    }

    #[test]
    fn a_lowered_module_answers_for_one_function_signature() {
        use zyntax_typed_ast::{Span, Type, TypedDeclaration, TypedFunction, TypedNode};
        let mut program = empty_program();
        for name in ["len", "abs", "len"] {
            let function = TypedFunction {
                name: zyntax_typed_ast::InternedString::new_global(name),
                return_type: Type::Unknown,
                body: Some(zyntax_typed_ast::TypedBlock {
                    statements: Vec::new(),
                    span: Span::default(),
                }),
                is_pure: name == "abs",
                ..Default::default()
            };
            program.declarations.push(TypedNode::new(
                TypedDeclaration::Function(function),
                Type::Unknown,
                Span::default(),
            ));
        }
        let mut arena = zyntax_typed_ast::AstArena::new();
        let hir = HirModule::new(arena.intern_string("lib"));
        let bytes = SnapshotBuilder::new("demo")
            .module_lowered("lib", program, &hir)
            .expect("module")
            .encode()
            .expect("encode");
        let snapshot = Snapshot::load_owned(bytes).expect("load");

        let abs = snapshot
            .function_signature("lib", "abs")
            .expect("decodes")
            .expect("declared");
        assert!(abs.is_pure);
        assert!(abs.body.is_none(), "a signature carries no body");
        let len = snapshot
            .function_signature("lib", "len")
            .expect("decodes")
            .expect("declared");
        assert!(!len.is_pure, "the first declaration under a name answers");
        assert_eq!(
            snapshot
                .declaration_signatures_at(0, "len")
                .expect("decodes all overloads")
                .len(),
            2,
            "declarations that share a name are all retained"
        );
        assert!(
            snapshot
                .function_signature("lib", "missing")
                .expect("decodes")
                .is_none()
        );
        assert!(
            snapshot
                .function_signature("other", "abs")
                .expect("decodes")
                .is_none()
        );
    }

    #[test]
    fn a_named_lowered_import_does_not_decode_the_whole_program() {
        use std::collections::HashSet;
        use zyntax_typed_ast::{Span, Type, TypedDeclaration, TypedFunction, TypedNode};

        let mut program = empty_program();
        for name in ["keep", "drop"] {
            program.declarations.push(TypedNode::new(
                TypedDeclaration::Function(TypedFunction {
                    name: zyntax_typed_ast::InternedString::new_global(name),
                    return_type: Type::Unknown,
                    body: Some(zyntax_typed_ast::TypedBlock {
                        statements: Vec::new(),
                        span: Span::default(),
                    }),
                    ..Default::default()
                }),
                Type::Unknown,
                Span::default(),
            ));
            if name == "keep" {
                program.declarations.push(TypedNode::new(
                    TypedDeclaration::Variable(zyntax_typed_ast::TypedVariable {
                        name: zyntax_typed_ast::InternedString::new_global("between"),
                        ty: Type::Primitive(zyntax_typed_ast::PrimitiveType::I64),
                        mutability: zyntax_typed_ast::Mutability::Immutable,
                        initializer: None,
                        visibility: zyntax_typed_ast::Visibility::Private,
                    }),
                    Type::Primitive(zyntax_typed_ast::PrimitiveType::I64),
                    Span::default(),
                ));
            }
        }
        let mut expected_program = program.clone();
        strip_bodies(&mut expected_program);
        let mut arena = zyntax_typed_ast::AstArena::new();
        let hir = HirModule::new(arena.intern_string("lib"));
        let bytes = SnapshotBuilder::new("demo")
            .module_lowered("lib", program, &hir)
            .expect("module")
            .encode()
            .expect("encode");
        let snapshot = Snapshot::load_owned(bytes).expect("load");

        let named = HashSet::from([zyntax_typed_ast::InternedString::new_global("keep")]);
        let selected = snapshot
            .module_selecting("lib", Some(&named))
            .expect("decodes")
            .expect("present");
        let expected = CompiledImport::new("demo", "lib", expected_program).selecting(&named);
        assert_eq!(
            selected.program().declarations,
            expected.program().declarations,
            "compact selection preserves complete nodes and declaration order"
        );
        let names: Vec<String> = selected
            .program()
            .declarations
            .iter()
            .filter_map(|decl| match &decl.node {
                TypedDeclaration::Function(function) => function.name.resolve_global(),
                _ => None,
            })
            .collect();
        assert_eq!(names, ["keep"]);
        assert!(selected.hir().is_some());
        assert!(snapshot.interfaces[0].get().is_some());
        assert!(
            snapshot.decoded[0].get().is_none(),
            "the complete typed program stays encoded"
        );
    }

    #[test]
    fn a_corrupted_snapshot_file_is_refused() {
        let mut arena = zyntax_typed_ast::AstArena::new();
        let hir = HirModule::new(arena.intern_string("lib"));
        let mut bytes = SnapshotBuilder::new("demo")
            .module_lowered("lib", empty_program(), &hir)
            .expect("module")
            .encode()
            .expect("encode");
        let snapshot = Snapshot::load_owned(bytes.clone()).expect("load");
        let extent = snapshot.modules[0].lowered.as_ref().expect("lowered").hir;
        let last = snapshot.blobs_at + (extent.at + extent.len) as usize - 1;
        bytes[last] = bytes[last].wrapping_add(1);
        // A snapshot from outside the executable is checked before its
        // HIR is read.
        let snapshot = Snapshot::load_owned(bytes).expect("the directory still reads");
        assert!(snapshot.module("lib").is_err());
    }

    fn compatibility_builder() -> SnapshotBuilder {
        let mut arena = zyntax_typed_ast::AstArena::new();
        let hir = HirModule::new(arena.intern_string("lib"));
        let mut program = empty_program();
        program.declarations.push(zyntax_typed_ast::TypedNode::new(
            zyntax_typed_ast::TypedDeclaration::Function(zyntax_typed_ast::TypedFunction {
                name: arena.intern_string("keep"),
                body: Some(zyntax_typed_ast::TypedBlock {
                    statements: vec![],
                    span: Default::default(),
                }),
                ..Default::default()
            }),
            zyntax_typed_ast::Type::Unknown,
            Default::default(),
        ));
        SnapshotBuilder::new("demo")
            .module_lowered("lib", program, &hir)
            .unwrap()
    }

    #[test]
    fn incompatible_hir_is_rejected_before_decoding_payloads() {
        for pointer_mismatch in [false, true] {
            let mut builder = compatibility_builder();
            let (_, build, pointer) = builder.modules[0].lowered.as_mut().unwrap();
            if pointer_mismatch {
                *pointer = 0;
            } else {
                build.push('x');
            }
            // The compatibility error takes precedence over malformed typed bodies.
            builder.modules[0].artifact = vec![0];
            let bytes = builder.encode().unwrap();
            let check = |result: Result<Snapshot, SnapshotError>| {
                let error = result.err().expect("incompatible HIR must be refused");
                if pointer_mismatch {
                    assert!(
                        matches!(&error,SnapshotError::IncompatiblePointerSize {module,found:0,expected} if module=="lib" && *expected==zyntax_compiler::target_pointer_size())
                    );
                } else {
                    assert!(
                        matches!(&error,SnapshotError::IncompatibleBuild {module,found,expected} if module=="lib" && found.ends_with('x') && *expected==zyntax_compiler::BUILD_ID)
                    );
                }
                assert!(error.to_string().contains("rebuild the snapshot"));
            };
            check(Snapshot::load_owned(bytes.clone()));
            check(Snapshot::load(Box::leak(bytes.into_boxed_slice())));
        }
    }

    #[test]
    fn cached_interfaces_and_bodies_cannot_bypass_compatibility_checks() {
        let mut snapshot = Snapshot::load_owned(compatibility_builder().encode().unwrap()).unwrap();
        snapshot.module("lib").unwrap().unwrap();
        snapshot.module_interface("lib").unwrap().unwrap();
        snapshot.function_signature("lib", "keep").unwrap().unwrap();
        snapshot.modules[0]
            .lowered
            .as_mut()
            .unwrap()
            .build_id
            .push('x');
        let named =
            std::collections::HashSet::from([zyntax_typed_ast::InternedString::new_global("keep")]);
        assert!(snapshot.validate_compatibility().is_err());
        assert!(!snapshot.carries_hir("lib"));
        assert!(snapshot.module("lib").is_err());
        assert!(snapshot.module_interface("lib").is_err());
        assert!(snapshot.function_signature("lib", "keep").is_err());
        assert!(snapshot.module_selecting("lib", Some(&named)).is_err());
        assert!(snapshot.hir_at(0).is_err());
        assert!(snapshot.modules().is_err());
        assert!(snapshot.into_modules().is_err());
    }

    #[cfg(feature = "native")]
    #[test]
    fn runtimes_reject_incompatible_snapshots_before_installing_a_grammar() {
        let mut snapshot =
            Snapshot::load_owned(compatibility_builder().grammar(vec![0]).encode().unwrap())
                .unwrap();
        snapshot.modules[0].lowered.as_mut().unwrap().pointer_size = 0;
        let snapshot = Arc::new(snapshot);
        let mut classic = crate::ZyntaxRuntime::new().unwrap();
        let error = classic
            .install_snapshot(Arc::clone(&snapshot))
            .err()
            .unwrap();
        assert!(error.to_string().contains("0-byte pointers"), "{error}");
        let mut tiered = crate::TieredRuntime::new(Default::default()).unwrap();
        let error = tiered.install_snapshot(snapshot).err().unwrap();
        assert!(error.to_string().contains("0-byte pointers"), "{error}");
        tiered.stop();
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
            Snapshot::load_owned(bytes),
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
        let cut = bytes[..bytes.len() - 32].to_vec();
        let snapshot = Snapshot::load_owned(cut).expect("the directory still reads");
        assert!(
            snapshot.module("prelude").is_err() || snapshot.grammar_bytes().is_none(),
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
        let snapshot = Snapshot::load_owned(bytes).expect("load");
        assert_eq!(snapshot.module_source("prelude"), Some("fn main() {}"));
        assert_eq!(snapshot.module_source("missing"), None);
    }
}
