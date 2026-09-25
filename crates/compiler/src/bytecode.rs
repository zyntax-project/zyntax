//! # Zyntax Bytecode Serialization
//!
//! This module provides serialization and deserialization of HIR to/from bytecode.
//!
//! ## Supported Formats
//!
//! - **Postcard (Binary)**: Compact, efficient binary format for production use
//! - **JSON**: Human-readable format for debugging and language interop
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zyntax_compiler::bytecode::{serialize_module, deserialize_module};
//! use zyntax_compiler::hir::HirModule;
//!
//! // Serialize to binary bytecode
//! let bytecode = serialize_module(&module, Format::Postcard)?;
//!
//! // Deserialize from bytecode
//! let module = deserialize_module(&bytecode, Format::Postcard)?;
//! ```

use crate::hir::{HirFunction, HirModule};
use std::borrow::Cow;
use std::io::{Read, Write};
use std::sync::OnceLock;
use thiserror::Error;

/// Bytecode serialization errors
#[derive(Error, Debug)]
pub enum BytecodeError {
    #[error("Serialization failed: {0}")]
    SerializationError(String),

    #[error("Deserialization failed: {0}")]
    DeserializationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid bytecode format")]
    InvalidFormat,

    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: String, actual: String },

    #[error("Checksum mismatch")]
    ChecksumMismatch,
}

pub type Result<T> = std::result::Result<T, BytecodeError>;

/// Bytecode serialization format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Postcard binary format (compact, fast)
    Postcard,
    /// JSON format (human-readable, for debugging)
    Json,
    /// Bincode format (alternative binary format)
    Bincode,
    /// Postcard, with each function body encoded on its own so a reader
    /// can decode only the bodies it reaches; see [`LazyModule`].
    Split,
}

/// A module whose functions stay encoded until asked for.
///
/// A program links against a library it reaches a little of; decoding
/// every function up front was most of what loading the library cost.
/// The image holds a directory of functions and, for each, a shell
/// (id, name, signature and attributes, no blocks) and a body, each
/// encoded on its own. Reading the image decodes the directory only;
/// [`Self::signature`] and [`Self::by_name`] decode one shell,
/// [`Self::function`] one body, and every id decoded is relocated by
/// the same base.
pub struct LazyModule {
    /// The image the extents address: borrowed from an image embedded
    /// in the executable, owned when it was read from anywhere else.
    bytes: Cow<'static, [u8]>,
    /// Where the bytes the extents address start in `bytes`.
    blob_at: usize,
    /// The module without its functions.
    stripped_at: Extent,
    /// Every function, in the module's order.
    directory: Vec<FnEntry>,
    /// Directory positions ordered by function id.
    by_id: Vec<u32>,
    /// Directory positions ordered by function name, ties in module order.
    by_name: Vec<u32>,
    /// Each function's shell, decoded when first asked for; boxed, so
    /// an empty slot costs a pointer.
    shells: Box<[OnceLock<Option<Box<HirFunction>>>]>,
    /// The module without its functions, decoded when first asked for.
    stripped: OnceLock<HirModule>,
    /// The module with a shell for each function, assembled when first
    /// asked for; set from the start for a module built in memory.
    whole: OnceLock<HirModule>,
    /// Whether the module was built in memory rather than read.
    in_memory: bool,
    /// What every id in the image is shifted by on decode.
    base: u32,
    /// Made on first use; see [`Self::link_index`].
    index: std::sync::OnceLock<std::sync::Arc<LinkIndex>>,
}

/// What a program linking a lowered module finds its functions and
/// globals by.
#[derive(Debug, Default)]
pub struct LinkIndex {
    /// Every function's id, by name.
    pub functions: std::collections::HashMap<zyntax_typed_ast::InternedString, crate::hir::HirId>,
    /// The names of the functions with a body.
    pub bodies: std::collections::HashSet<zyntax_typed_ast::InternedString>,
    /// Every global's id, by name.
    pub globals: std::collections::HashMap<zyntax_typed_ast::InternedString, crate::hir::HirId>,
    /// The boxed-constant initializers: called by the host rather than
    /// any body, and adopted with the first function a program reaches,
    /// since their boxes are what the bodies load.
    pub inits: Vec<crate::hir::HirId>,
}

impl LinkIndex {
    /// Read from the module's directory and its function-less part;
    /// no function of a module that was read is decoded.
    fn of(module: &LazyModule) -> Self {
        let mut index = Self::default();
        if module.in_memory {
            let module = module.shell();
            for f in module.functions.values() {
                index.functions.insert(f.name, f.id);
                if !f.is_external {
                    index.bodies.insert(f.name);
                }
                if f.name
                    .resolve_global()
                    .is_some_and(|n| crate::const_boxes::is_init_function(&n))
                {
                    index.inits.push(f.id);
                }
            }
            for g in module.globals.values() {
                index.globals.insert(g.name, g.id);
            }
            return index;
        }
        module.for_each_function(|name, id, has_body| {
            let interned = zyntax_typed_ast::InternedString::new_global(name);
            index.functions.insert(interned, id);
            if has_body {
                index.bodies.insert(interned);
            }
            if crate::const_boxes::is_init_function(name) {
                index.inits.push(id);
            }
        });
        for g in module.stripped().globals.values() {
            index.globals.insert(g.name, g.id);
        }
        index
    }

    /// Several modules' indexes as one; a later module's name wins.
    pub fn merged<'a>(indexes: impl IntoIterator<Item = &'a LinkIndex>) -> Self {
        let mut all = Self::default();
        for index in indexes {
            all.functions
                .extend(index.functions.iter().map(|(k, v)| (*k, *v)));
            all.bodies.extend(index.bodies.iter().copied());
            all.globals
                .extend(index.globals.iter().map(|(k, v)| (*k, *v)));
            all.inits.extend(index.inits.iter().copied());
        }
        all
    }
}

impl std::fmt::Debug for LazyModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyModule")
            .field("image_len", &self.bytes.len())
            .field("functions", &self.directory.len())
            .field("in_memory", &self.in_memory)
            .field("base", &self.base)
            .finish()
    }
}

/// Where something sits among the bytes after a [`SplitPayload`].
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug)]
struct Extent {
    at: u32,
    len: u32,
}

/// One function of a [`Format::Split`] image.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug)]
struct FnEntry {
    id: crate::hir::HirId,
    /// The function's name, UTF-8.
    name: Extent,
    /// The function without blocks, values or locals; an external
    /// function whole.
    shell: Extent,
    /// The whole function; absent for an external one.
    body: Option<Extent>,
}

/// The wire shape of [`Format::Split`]: this, then `blob_len` bytes
/// that its extents address.
#[derive(serde::Serialize, serde::Deserialize)]
struct SplitPayload {
    /// The module with no functions.
    stripped: Extent,
    directory: Vec<FnEntry>,
    by_id: Vec<u32>,
    by_name: Vec<u32>,
    blob_len: u32,
    /// The largest id anywhere in the module, before relocation.
    max_id: u32,
}

impl LazyModule {
    /// A module already in memory: nothing to decode later.
    pub fn eager(module: HirModule) -> Self {
        Self {
            bytes: Cow::Borrowed(&[]),
            blob_at: 0,
            stripped_at: Extent { at: 0, len: 0 },
            directory: Vec::new(),
            by_id: Vec::new(),
            by_name: Vec::new(),
            shells: Box::new([]),
            stripped: OnceLock::new(),
            whole: OnceLock::from(module),
            in_memory: true,
            base: 0,
            index: std::sync::OnceLock::new(),
        }
    }

    /// The module's functions and globals by name, made once.
    pub fn link_index(&self) -> std::sync::Arc<LinkIndex> {
        std::sync::Arc::clone(
            self.index
                .get_or_init(|| std::sync::Arc::new(LinkIndex::of(self))),
        )
    }

    fn blob(&self, extent: Extent) -> Option<&[u8]> {
        let at = self.blob_at.checked_add(extent.at as usize)?;
        self.bytes.get(at..at.checked_add(extent.len as usize)?)
    }

    fn decode<T: serde::de::DeserializeOwned>(&self, extent: Extent) -> Option<T> {
        let bytes = self.blob(extent)?;
        crate::hir::HirId::relocated_by(self.base, || postcard::from_bytes(bytes).ok())
    }

    /// The directory position of function `id`.
    fn position(&self, id: crate::hir::HirId) -> Option<usize> {
        let at = self
            .by_id
            .binary_search_by_key(&id.as_u32(), |&i| {
                self.directory
                    .get(i as usize)
                    .map_or(u32::MAX, |e| e.id.as_u32())
            })
            .ok()?;
        Some(self.by_id[at] as usize)
    }

    fn name_at(&self, i: usize) -> &[u8] {
        self.directory
            .get(i)
            .and_then(|e| self.blob(e.name))
            .unwrap_or_default()
    }

    fn shell_at(&self, i: usize) -> Option<&HirFunction> {
        self.shells[i]
            .get_or_init(|| self.decode(self.directory[i].shell))
            .as_deref()
    }

    fn function_at(&self, i: usize) -> Option<HirFunction> {
        match self.directory[i].body {
            Some(body) => self.decode(body),
            None => self.shell_at(i).cloned(),
        }
    }

    /// The module without its functions: globals, types, effects.
    /// A module built in memory answers with itself.
    pub fn stripped(&self) -> &HirModule {
        if let Some(whole) = self.in_memory.then(|| self.whole.get()).flatten() {
            return whole;
        }
        self.stripped.get_or_init(|| {
            self.decode(self.stripped_at)
                .expect("a split module image decodes: its checksum or its build vouched for it")
        })
    }

    /// The module with a shell for each function: globals, types,
    /// externs, and for each function with a body its id, name,
    /// signature and attributes and no blocks. Decodes every shell;
    /// [`Self::signature`] and [`Self::by_name`] decode one.
    pub fn shell(&self) -> &HirModule {
        self.whole.get_or_init(|| {
            let mut module = self.stripped().clone();
            for i in 0..self.directory.len() {
                let shell = self.shell_at(i).expect(
                    "a split module image decodes: its checksum or its build vouched for it",
                );
                module.functions.insert(shell.id, shell.clone());
            }
            module
        })
    }

    /// Call `each` with every function's name and id, in the module's
    /// order, and whether the function has a body here. Decodes no
    /// function of a module that was read.
    pub fn for_each_function(&self, mut each: impl FnMut(&str, crate::hir::HirId, bool)) {
        if self.in_memory {
            for function in self.shell().functions.values() {
                let name = function.name.resolve_global().unwrap_or_default();
                each(&name, function.id, !function.is_external);
            }
            return;
        }
        for (i, entry) in self.directory.iter().enumerate() {
            let name = std::str::from_utf8(self.name_at(i)).unwrap_or_default();
            each(name, entry.id, entry.body.is_some());
        }
    }

    /// How many functions the module holds.
    pub fn function_count(&self) -> usize {
        if self.in_memory {
            return self.shell().functions.len();
        }
        self.directory.len()
    }

    /// Whether `id` names a function of this module.
    pub fn has_function(&self, id: crate::hir::HirId) -> bool {
        if self.in_memory {
            return self.shell().functions.contains_key(&id);
        }
        self.position(id).is_some()
    }

    /// Function `id` without its body: signature and attributes. For a
    /// module built in memory, the function itself.
    pub fn signature(&self, id: crate::hir::HirId) -> Option<&HirFunction> {
        if self.in_memory {
            return self.shell().functions.get(&id);
        }
        self.shell_at(self.position(id)?)
    }

    /// The function named `name`, as [`Self::signature`] gives it; the
    /// first in module order when several share it.
    pub fn by_name(&self, name: &str) -> Option<&HirFunction> {
        if self.in_memory {
            return self
                .shell()
                .functions
                .values()
                .find(|f| f.name.resolve_global().as_deref() == Some(name));
        }
        let first = self
            .by_name
            .partition_point(|&i| self.name_at(i as usize) < name.as_bytes());
        let &i = self.by_name.get(first)?;
        (self.name_at(i as usize) == name.as_bytes())
            .then(|| self.shell_at(i as usize))
            .flatten()
    }

    /// The function `id`, its body decoded.
    pub fn function(&self, id: crate::hir::HirId) -> Option<HirFunction> {
        if self.in_memory {
            return self.shell().functions.get(&id).cloned();
        }
        self.function_at(self.position(id)?)
    }

    /// Every function, decoded, in the module's order.
    pub fn functions(&self) -> Box<dyn Iterator<Item = (crate::hir::HirId, HirFunction)> + '_> {
        if self.in_memory {
            return Box::new(
                self.shell()
                    .functions
                    .iter()
                    .map(|(id, f)| (*id, f.clone())),
            );
        }
        Box::new(
            (0..self.directory.len())
                .filter_map(move |i| self.function_at(i).map(|f| (self.directory[i].id, f))),
        )
    }

    /// The whole module, every body decoded.
    pub fn into_module(mut self) -> HirModule {
        if self.in_memory {
            return self
                .whole
                .take()
                .expect("a module built in memory is whole from the start");
        }
        let mut module = self.stripped().clone();
        for i in 0..self.directory.len() {
            if let Some(function) = self.function_at(i) {
                module.functions.insert(self.directory[i].id, function);
            }
        }
        module
    }
}

/// Serialize a module as [`Format::Split`].
pub fn serialize_module_split(module: &HirModule) -> Result<Vec<u8>> {
    let mut blob: Vec<u8> = Vec::new();
    let mut put = |bytes: &[u8]| -> Result<Extent> {
        let extent = Extent {
            at: u32::try_from(blob.len())
                .map_err(|e| BytecodeError::SerializationError(e.to_string()))?,
            len: u32::try_from(bytes.len())
                .map_err(|e| BytecodeError::SerializationError(e.to_string()))?,
        };
        blob.extend_from_slice(bytes);
        Ok(extent)
    };

    let mut stripped = module.clone();
    stripped.functions.clear();
    let stripped = put(&to_postcard(&stripped)?)?;

    let mut directory = Vec::with_capacity(module.functions.len());
    let mut names = Vec::with_capacity(module.functions.len());
    for (id, function) in &module.functions {
        let name = function.name.resolve_global().unwrap_or_default();
        let name_extent = put(name.as_bytes())?;
        let (shell, body) = if function.is_external {
            (put(&to_postcard(function)?)?, None)
        } else {
            let shell = HirFunction {
                blocks: Default::default(),
                locals: Default::default(),
                values: Default::default(),
                ..function.clone()
            };
            (
                put(&to_postcard(&shell)?)?,
                Some(put(&to_postcard(function)?)?),
            )
        };
        names.push(name);
        directory.push(FnEntry {
            id: *id,
            name: name_extent,
            shell,
            body,
        });
    }
    let positions = 0..directory.len() as u32;
    let mut by_id: Vec<u32> = positions.clone().collect();
    by_id.sort_by_key(|&i| directory[i as usize].id.as_u32());
    let mut by_name: Vec<u32> = positions.collect();
    by_name.sort_by(|&a, &b| names[a as usize].cmp(&names[b as usize]));

    let payload = SplitPayload {
        stripped,
        directory,
        by_id,
        by_name,
        blob_len: u32::try_from(blob.len())
            .map_err(|e| BytecodeError::SerializationError(e.to_string()))?,
        max_id: max_hir_id(module),
    };
    let mut bytes = to_postcard(&payload)?;
    bytes.extend_from_slice(&blob);
    Ok(with_header(module, Format::Split, bytes))
}

fn to_postcard<T: serde::Serialize + ?Sized>(value: &T) -> Result<Vec<u8>> {
    postcard::to_allocvec(value).map_err(|e| BytecodeError::SerializationError(e.to_string()))
}

/// Read a [`Format::Split`] image without decoding its functions,
/// checking the image against its checksum first. Any other format is
/// decoded whole and wrapped.
pub fn deserialize_module_lazy(bytes: impl Into<Cow<'static, [u8]>>) -> Result<LazyModule> {
    lazy_module(bytes.into(), true)
}

/// [`deserialize_module_lazy`] without the checksum, for an image this
/// build embedded in its own executable. An image from anywhere else
/// goes through the checked read.
pub fn deserialize_module_lazy_trusted(bytes: &'static [u8]) -> Result<LazyModule> {
    lazy_module(Cow::Borrowed(bytes), false)
}

fn lazy_module(bytes: Cow<'static, [u8]>, checked: bool) -> Result<LazyModule> {
    let (header, payload) = if checked {
        checked_payload(&bytes)?
    } else {
        header_and_payload(&bytes)?
    };
    if Format::from_u8(header.format)? != Format::Split {
        return deserialize_module(&bytes).map(LazyModule::eager);
    }
    let base = crate::hir::HirId::next_unminted();
    let (split, blob): (SplitPayload, &[u8]) = crate::hir::HirId::relocated_by(base, || {
        postcard::take_from_bytes(payload)
            .map_err(|e| BytecodeError::DeserializationError(e.to_string()))
    })?;
    let n = split.directory.len();
    if blob.len() != split.blob_len as usize || split.by_id.len() != n || split.by_name.len() != n {
        return Err(BytecodeError::InvalidFormat);
    }
    let blob_at = bytes.len() - blob.len();
    crate::hir::HirId::ensure_counter_above(base.saturating_add(split.max_id));
    Ok(LazyModule {
        bytes,
        blob_at,
        stripped_at: split.stripped,
        shells: (0..n).map(|_| OnceLock::new()).collect(),
        directory: split.directory,
        by_id: split.by_id,
        by_name: split.by_name,
        stripped: OnceLock::new(),
        whole: OnceLock::new(),
        in_memory: false,
        base,
        index: std::sync::OnceLock::new(),
    })
}

/// Bytecode file header
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BytecodeHeader {
    /// Magic number 0x5A424300, written little-endian: the file starts
    /// with the bytes 00 43 42 5A ("\0CBZ").
    pub magic: u32,
    /// Major version
    pub major_version: u16,
    /// Minor version
    pub minor_version: u16,
    /// Format used for payload
    pub format: u8,
    /// Flags bitfield
    pub flags: u32,
    /// Module UUID
    pub module_id: uuid::Uuid,
    /// Payload size in bytes
    pub payload_size: u64,
    /// CRC32 checksum of payload
    pub checksum: u32,
}

impl BytecodeHeader {
    const MAGIC: u32 = 0x5A424300; // on disk, little-endian: 00 43 42 5A ("\0CBZ")
    // Moves with any change to a payload's layout, so a payload in an
    // older layout is refused with VersionMismatch and a cache loader
    // recompiles rather than misreading it.
    const CURRENT_MAJOR: u16 = 3;
    const CURRENT_MINOR: u16 = 0;

    /// Create a new header for the given module and format
    pub fn new(module: &HirModule, format: Format) -> Self {
        // The on-disk header reserves a 16-byte slot historically named
        // `module_id` (UUID). HirId is now u32; we zero-extend it into
        // those 16 bytes so the wire format stays fixed-width.
        let mut id_bytes = [0u8; 16];
        id_bytes[..4].copy_from_slice(&module.id.as_u32().to_le_bytes());
        Self {
            magic: Self::MAGIC,
            major_version: Self::CURRENT_MAJOR,
            minor_version: Self::CURRENT_MINOR,
            format: format.to_u8(),
            flags: 0,
            module_id: uuid::Uuid::from_bytes(id_bytes),
            payload_size: 0, // Will be filled in during serialization
            checksum: 0,     // Will be calculated during serialization
        }
    }

    /// Validate the header
    pub fn validate(&self) -> Result<()> {
        if self.magic != Self::MAGIC {
            return Err(BytecodeError::InvalidFormat);
        }

        if self.major_version != Self::CURRENT_MAJOR {
            return Err(BytecodeError::VersionMismatch {
                expected: format!("{}.x", Self::CURRENT_MAJOR),
                actual: format!("{}.{}", self.major_version, self.minor_version),
            });
        }

        Ok(())
    }
}

impl Format {
    fn to_u8(self) -> u8 {
        match self {
            Format::Postcard => 0,
            Format::Json => 1,
            Format::Bincode => 2,
            Format::Split => 3,
        }
    }

    fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Format::Postcard),
            1 => Ok(Format::Json),
            2 => Ok(Format::Bincode),
            3 => Ok(Format::Split),
            _ => Err(BytecodeError::InvalidFormat),
        }
    }
}

// (The old AsUuid helper trait is gone; HirId is now u32 and we encode
// it directly into the fixed-width header slot in `BytecodeHeader::new`.)

/// Serialize header to raw 44-byte format (matches deserialize_raw_header)
fn serialize_raw_header(header: &BytecodeHeader) -> Vec<u8> {
    const HEADER_SIZE: usize = 44;
    let mut bytes = Vec::with_capacity(HEADER_SIZE);

    // magic (u32 little-endian)
    bytes.extend_from_slice(&header.magic.to_le_bytes());
    // major_version (u16 little-endian)
    bytes.extend_from_slice(&header.major_version.to_le_bytes());
    // minor_version (u16 little-endian)
    bytes.extend_from_slice(&header.minor_version.to_le_bytes());
    // format (u8)
    bytes.push(header.format);
    // padding (3 bytes)
    bytes.extend_from_slice(&[0u8; 3]);
    // flags (u32 little-endian)
    bytes.extend_from_slice(&header.flags.to_le_bytes());
    // module_id (16 bytes UUID)
    bytes.extend_from_slice(header.module_id.as_bytes());
    // payload_size (u64 little-endian)
    bytes.extend_from_slice(&header.payload_size.to_le_bytes());
    // checksum (u32 little-endian)
    bytes.extend_from_slice(&header.checksum.to_le_bytes());

    debug_assert_eq!(bytes.len(), HEADER_SIZE);
    bytes
}

/// Serialize a HIR module to bytecode
pub fn serialize_module(module: &HirModule, format: Format) -> Result<Vec<u8>> {
    // Serialize the module payload
    let payload = match format {
        Format::Postcard => postcard::to_allocvec(module)
            .map_err(|e| BytecodeError::SerializationError(e.to_string()))?,
        Format::Json => serde_json::to_vec_pretty(module)
            .map_err(|e| BytecodeError::SerializationError(e.to_string()))?,
        Format::Bincode => bincode::serialize(module)
            .map_err(|e| BytecodeError::SerializationError(e.to_string()))?,
        Format::Split => return serialize_module_split(module),
    };
    Ok(with_header(module, format, payload))
}

/// The header for `payload`, then the payload.
fn with_header(module: &HirModule, format: Format, payload: Vec<u8>) -> Vec<u8> {
    let checksum = crc32fast::hash(&payload);
    let mut header = BytecodeHeader::new(module, format);
    header.payload_size = payload.len() as u64;
    header.checksum = checksum;
    let header_bytes = serialize_raw_header(&header);
    let mut result = Vec::with_capacity(header_bytes.len() + payload.len());
    result.extend_from_slice(&header_bytes);
    result.extend_from_slice(&payload);
    result
}

/// The header and the payload it covers, once the header has been
/// validated and the payload's checksum has been verified.
fn checked_payload(bytes: &[u8]) -> Result<(BytecodeHeader, &[u8])> {
    let (header, payload) = header_and_payload(bytes)?;
    if crc32fast::hash(payload) != header.checksum {
        return Err(BytecodeError::ChecksumMismatch);
    }
    Ok((header, payload))
}

/// The header and the payload it covers, once the header has been
/// validated; the payload is not checked against its checksum.
fn header_and_payload(bytes: &[u8]) -> Result<(BytecodeHeader, &[u8])> {
    const HEADER_SIZE: usize = 44;
    if bytes.len() < HEADER_SIZE {
        return Err(BytecodeError::InvalidFormat);
    }
    let (header, header_size) = deserialize_raw_header(bytes)?;
    header.validate()?;
    Ok((header, &bytes[header_size..]))
}

/// Serialize a HIR module to a writer
pub fn serialize_module_to_writer<W: Write>(
    module: &HirModule,
    format: Format,
    writer: &mut W,
) -> Result<()> {
    let bytes = serialize_module(module, format)?;
    writer.write_all(&bytes)?;
    Ok(())
}

/// Serialize a HIR module to a file
pub fn serialize_module_to_file(
    module: &HirModule,
    format: Format,
    path: &std::path::Path,
) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    serialize_module_to_writer(module, format, &mut file)
}

/// Deserialize raw header (44 bytes fixed format, no bincode)
fn deserialize_raw_header(bytes: &[u8]) -> Result<(BytecodeHeader, usize)> {
    const HEADER_SIZE: usize = 44;
    if bytes.len() < HEADER_SIZE {
        return Err(BytecodeError::InvalidFormat);
    }

    let mut cursor = std::io::Cursor::new(bytes);
    use std::io::Read;

    // Read magic (u32 little-endian)
    let mut buf4 = [0u8; 4];
    cursor.read_exact(&mut buf4)?;
    let magic = u32::from_le_bytes(buf4);

    // Early validation: if magic doesn't match, this isn't raw format
    if magic != BytecodeHeader::MAGIC {
        return Err(BytecodeError::InvalidFormat);
    }

    // Read major_version (u16 little-endian)
    let mut buf2 = [0u8; 2];
    cursor.read_exact(&mut buf2)?;
    let major_version = u16::from_le_bytes(buf2);

    // Read minor_version (u16 little-endian)
    cursor.read_exact(&mut buf2)?;
    let minor_version = u16::from_le_bytes(buf2);

    // Read format (u8)
    let mut buf1 = [0u8; 1];
    cursor.read_exact(&mut buf1)?;
    let format = buf1[0];

    // Skip padding (3 bytes)
    let mut padding = [0u8; 3];
    cursor.read_exact(&mut padding)?;

    // Read flags (u32 little-endian)
    cursor.read_exact(&mut buf4)?;
    let flags = u32::from_le_bytes(buf4);

    // Read module_id (16 bytes UUID)
    let mut uuid_bytes = [0u8; 16];
    cursor.read_exact(&mut uuid_bytes)?;
    let module_id = uuid::Uuid::from_bytes(uuid_bytes);

    // Read payload_size (u64 little-endian)
    let mut buf8 = [0u8; 8];
    cursor.read_exact(&mut buf8)?;
    let payload_size = u64::from_le_bytes(buf8);

    // Read checksum (u32 little-endian)
    cursor.read_exact(&mut buf4)?;
    let checksum = u32::from_le_bytes(buf4);

    let header = BytecodeHeader {
        magic,
        major_version,
        minor_version,
        format,
        flags,
        module_id,
        payload_size,
        checksum,
    };

    Ok((header, HEADER_SIZE))
}

/// Deserialize a HIR module from bytecode
pub fn deserialize_module(bytes: &[u8]) -> Result<HirModule> {
    let (header, payload) = checked_payload(bytes)?;
    let format = Format::from_u8(header.format)?;
    if format == Format::Split {
        return lazy_module(Cow::Owned(bytes.to_vec()), false).map(LazyModule::into_module);
    }

    // Deserialize payload. The module's ids were minted by whichever
    // process wrote it and may already be in use here, so every id read
    // is shifted above the next unminted one; the counter is then moved
    // past the highest id the module holds, so later `HirId::new()`
    // calls cannot land on one of them either.
    let module: HirModule =
        crate::hir::HirId::relocated_by(crate::hir::HirId::next_unminted(), || match format {
            Format::Postcard => postcard::from_bytes(payload)
                .map_err(|e| BytecodeError::DeserializationError(e.to_string())),
            Format::Json => serde_json::from_slice(payload)
                .map_err(|e| BytecodeError::DeserializationError(e.to_string())),
            Format::Bincode => bincode::deserialize(payload)
                .map_err(|e| BytecodeError::DeserializationError(e.to_string())),
            Format::Split => unreachable!("handled above"),
        })?;
    advance_hir_id_counter(&module);

    Ok(module)
}

/// Bump the global `HirId` counter above every id defined in `module`.
/// See [`HirId::ensure_counter_above`].
fn advance_hir_id_counter(module: &HirModule) {
    crate::hir::HirId::ensure_counter_above(max_hir_id(module));
}

/// The largest id anywhere in `module`.
fn max_hir_id(module: &HirModule) -> u32 {
    let mut max_id = module.id.as_u32();
    for func in module.functions.values() {
        max_id = max_id.max(func.id.as_u32());
        for id in func.values.keys() {
            max_id = max_id.max(id.as_u32());
        }
        for id in func.blocks.keys() {
            max_id = max_id.max(id.as_u32());
        }
        for id in func.locals.keys() {
            max_id = max_id.max(id.as_u32());
        }
        for param in &func.signature.params {
            max_id = max_id.max(param.id.as_u32());
        }
    }
    for id in module.globals.keys() {
        max_id = max_id.max(id.as_u32());
    }
    for id in module.effects.keys() {
        max_id = max_id.max(id.as_u32());
    }
    for id in module.handlers.keys() {
        max_id = max_id.max(id.as_u32());
    }
    max_id
}

/// Deserialize a HIR module from a reader
pub fn deserialize_module_from_reader<R: Read>(reader: &mut R) -> Result<HirModule> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    deserialize_module(&bytes)
}

/// Deserialize a HIR module from a file
pub fn deserialize_module_from_file(path: &std::path::Path) -> Result<HirModule> {
    let mut file = std::fs::File::open(path)?;
    deserialize_module_from_reader(&mut file)
}

/// Get bytecode statistics
pub fn bytecode_stats(bytes: &[u8]) -> Result<BytecodeStats> {
    const HEADER_SIZE: usize = 44;
    if bytes.len() < HEADER_SIZE {
        return Err(BytecodeError::InvalidFormat);
    }

    // Use raw header format (matches serialize_raw_header)
    let (header, header_size) = deserialize_raw_header(bytes)?;

    header.validate()?;

    Ok(BytecodeStats {
        total_size: bytes.len(),
        header_size,
        payload_size: header.payload_size as usize,
        format: Format::from_u8(header.format)?,
        version: format!("{}.{}", header.major_version, header.minor_version),
        module_id: header.module_id,
    })
}

/// Bytecode statistics
#[derive(Debug, Clone)]
pub struct BytecodeStats {
    pub total_size: usize,
    pub header_size: usize,
    pub payload_size: usize,
    pub format: Format,
    pub version: String,
    pub module_id: uuid::Uuid,
}

impl std::fmt::Display for BytecodeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Bytecode Stats:\n  Total Size: {} bytes\n  Header Size: {} bytes\n  Payload Size: {} bytes\n  Format: {:?}\n  Version: {}\n  Module ID: {}",
            self.total_size,
            self.header_size,
            self.payload_size,
            self.format,
            self.version,
            self.module_id
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::*;
    use indexmap::IndexMap;
    use zyntax_typed_ast::InternedString;

    fn create_test_module() -> HirModule {
        // Create a temporary arena for string interning
        let mut arena = zyntax_typed_ast::AstArena::new();
        let name = arena.intern_string("test_module");

        HirModule {
            id: HirId::new(),
            name,
            functions: IndexMap::new(),
            globals: IndexMap::new(),
            types: IndexMap::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            version: 1,
            dependencies: std::collections::HashSet::new(),
            effects: IndexMap::new(),
            handlers: IndexMap::new(),
            automatic_release: false,
        }
    }

    #[test]
    fn test_serialize_deserialize_postcard() {
        let module = create_test_module();
        let bytecode = serialize_module(&module, Format::Postcard).unwrap();
        let deserialized = deserialize_module(&bytecode).unwrap();

        assert_eq!(module.name, deserialized.name);
        assert_eq!(module.version, deserialized.version);
    }

    #[test]
    fn deserialize_advances_hir_id_counter() {
        // Regression: deserialize rebuilds ids via `from_raw` without
        // touching the global counter. If the counter isn't advanced, a
        // later `HirId::new()` can collide with a loaded id and overwrite
        // it in the values map — the fib bench-cache crash, where a new
        // instruction result reused a parameter's id and dropped the
        // parameter. After loading, the next minted id must be past the
        // module's maximum id.
        let mut module = create_test_module();
        let base = HirId::new().as_u32();
        let big = base + 100_000;

        let mut arena = zyntax_typed_ast::AstArena::new();
        let fname = arena.intern_string("f");
        let sig = HirFunctionSignature {
            params: vec![],
            returns: vec![],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(fname, sig);
        func.id = HirId::from_raw(big);
        func.values.insert(
            HirId::from_raw(big),
            HirValue {
                id: HirId::from_raw(big),
                ty: HirType::I64,
                kind: HirValueKind::Parameter(0),
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );
        module.functions.insert(func.id, func);

        let bytes = serialize_module(&module, Format::Postcard).unwrap();
        let _ = deserialize_module(&bytes).unwrap();

        let next = HirId::new().as_u32();
        assert!(
            next > big,
            "deserialize must advance the id counter past the module max ({big}); got {next}"
        );
    }

    #[test]
    fn a_loaded_module_lands_above_every_id_minted_here() {
        // A module carries the ids its own process minted, and this
        // process may have minted the same numbers already. Loading
        // shifts every id in the module together, so a function still
        // finds its own values and the counter keeps clear of them.
        let mut module = create_test_module();
        let mut arena = zyntax_typed_ast::AstArena::new();
        let sig = HirFunctionSignature {
            params: vec![],
            returns: vec![],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(arena.intern_string("f"), sig);
        func.id = HirId::from_raw(1);
        func.entry_block = HirId::from_raw(2);
        func.values.insert(
            HirId::from_raw(3),
            HirValue {
                id: HirId::from_raw(3),
                ty: HirType::I64,
                kind: HirValueKind::Parameter(0),
                uses: std::collections::HashSet::new(),
                span: None,
            },
        );
        module.functions.insert(func.id, func);

        let bytes = serialize_module(&module, Format::Postcard).unwrap();
        let floor = HirId::new().as_u32();
        let loaded = deserialize_module(&bytes).unwrap();

        let (id, function) = loaded.functions.iter().next().unwrap();
        assert!(id.as_u32() > floor, "function id {id:?} not above {floor}");
        assert_eq!(*id, function.id, "the key moved with the function");
        assert_eq!(
            function.entry_block.as_u32(),
            id.as_u32() + 1,
            "references shift by the same amount"
        );
        let (value_id, value) = function.values.iter().next().unwrap();
        assert_eq!(*value_id, value.id);
        assert_eq!(value_id.as_u32(), id.as_u32() + 2);
        assert!(HirId::new().as_u32() > value_id.as_u32());
    }

    fn empty_signature() -> HirFunctionSignature {
        HirFunctionSignature {
            params: vec![],
            returns: vec![],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: false,
        }
    }

    /// A module with two functions that have bodies and an external one.
    fn split_test_module() -> HirModule {
        let mut module = create_test_module();
        let mut arena = zyntax_typed_ast::AstArena::new();
        for name in ["zeta", "alpha"] {
            let mut func = HirFunction::new(arena.intern_string(name), empty_signature());
            let value = HirId::new();
            func.values.insert(
                value,
                HirValue {
                    id: value,
                    ty: HirType::I64,
                    kind: HirValueKind::Parameter(0),
                    uses: std::collections::HashSet::new(),
                    span: None,
                },
            );
            module.functions.insert(func.id, func);
        }
        let mut external = HirFunction::new(arena.intern_string("puts"), empty_signature());
        external.is_external = true;
        external.blocks.clear();
        module.functions.insert(external.id, external);
        module
    }

    fn name_of(function: &HirFunction) -> String {
        function.name.resolve_global().unwrap_or_default()
    }

    #[test]
    fn a_split_module_decodes_one_function_at_a_time() {
        let module = split_test_module();
        let bytes = serialize_module(&module, Format::Split).unwrap();
        let lazy = deserialize_module_lazy(bytes).unwrap();
        assert_eq!(lazy.function_count(), 3);

        let mut names = Vec::new();
        lazy.for_each_function(|name, _, has_body| names.push((name.to_string(), has_body)));
        assert_eq!(
            names,
            vec![
                ("zeta".to_string(), true),
                ("alpha".to_string(), true),
                ("puts".to_string(), false)
            ],
            "the directory keeps the module's order"
        );

        let alpha = lazy.by_name("alpha").expect("found by name");
        assert_eq!(name_of(alpha), "alpha");
        assert!(
            alpha.blocks.is_empty() && alpha.values.is_empty(),
            "a shell has no body"
        );
        assert!(lazy.by_name("beta").is_none());

        let id = alpha.id;
        assert!(lazy.has_function(id));
        assert_eq!(lazy.signature(id).map(|f| f.id), Some(id));
        let body = lazy.function(id).expect("the body decodes");
        assert_eq!(body.id, id, "the body relocates by the shell's base");
        let (value_id, value) = body.values.iter().next().expect("a value");
        assert_eq!(*value_id, value.id);
        assert!(value_id.as_u32() > id.as_u32());
        assert!(HirId::new().as_u32() > value_id.as_u32());

        let puts = lazy.by_name("puts").expect("an external function");
        assert!(puts.is_external);
        assert_eq!(lazy.function(puts.id).map(|f| f.id), Some(puts.id));

        assert_eq!(lazy.shell().functions.len(), 3);
        assert_eq!(lazy.functions().count(), 3);
        let whole = lazy.into_module();
        assert_eq!(whole.functions.len(), 3);
        let names: Vec<String> = whole.functions.values().map(name_of).collect();
        assert_eq!(names, ["zeta", "alpha", "puts"]);
    }

    #[test]
    fn a_split_module_decodes_whole() {
        let module = split_test_module();
        let bytes = serialize_module(&module, Format::Split).unwrap();
        let whole = deserialize_module(&bytes).unwrap();
        assert_eq!(whole.functions.len(), 3);
        assert!(
            whole
                .functions
                .values()
                .filter(|f| !f.is_external)
                .all(|f| f.values.len() == 1),
            "every body arrives"
        );
    }

    #[test]
    fn an_embedded_split_module_reads_without_its_checksum() {
        let bytes = serialize_module(&split_test_module(), Format::Split).unwrap();
        let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let lazy = deserialize_module_lazy_trusted(bytes).unwrap();
        let alpha = lazy.by_name("alpha").expect("found by name").id;
        assert!(lazy.function(alpha).is_some());
    }

    #[test]
    fn a_corrupted_split_module_fails_the_checked_read() {
        let mut bytes = serialize_module(&split_test_module(), Format::Split).unwrap();
        let last = bytes.len() - 1;
        bytes[last] = bytes[last].wrapping_add(1);
        assert!(matches!(
            deserialize_module_lazy(bytes),
            Err(BytecodeError::ChecksumMismatch)
        ));
    }

    #[test]
    fn a_payload_in_an_older_layout_is_refused_by_version() {
        let mut bytes = serialize_module(&split_test_module(), Format::Split).unwrap();
        // The major version follows the four magic bytes.
        let older = BytecodeHeader::CURRENT_MAJOR - 1;
        bytes[4..6].copy_from_slice(&older.to_le_bytes());
        assert!(matches!(
            deserialize_module_lazy(bytes.clone()),
            Err(BytecodeError::VersionMismatch { .. })
        ));
        assert!(matches!(
            deserialize_module(&bytes),
            Err(BytecodeError::VersionMismatch { .. })
        ));
    }

    #[test]
    fn test_serialize_deserialize_json() {
        let module = create_test_module();
        let bytecode = serialize_module(&module, Format::Json).unwrap();
        let deserialized = deserialize_module(&bytecode).unwrap();

        assert_eq!(module.name, deserialized.name);
        assert_eq!(module.version, deserialized.version);
    }

    #[test]
    fn test_serialize_deserialize_bincode() {
        let module = create_test_module();
        let bytecode = serialize_module(&module, Format::Bincode).unwrap();
        let deserialized = deserialize_module(&bytecode).unwrap();

        assert_eq!(module.name, deserialized.name);
        assert_eq!(module.version, deserialized.version);
    }

    #[test]
    fn test_checksum_validation() {
        let module = create_test_module();
        let mut bytecode = serialize_module(&module, Format::Postcard).unwrap();

        // Corrupt the payload
        if let Some(byte) = bytecode.last_mut() {
            *byte = byte.wrapping_add(1);
        }

        let result = deserialize_module(&bytecode);
        assert!(matches!(result, Err(BytecodeError::ChecksumMismatch)));
    }

    #[test]
    fn test_bytecode_stats() {
        let module = create_test_module();
        let bytecode = serialize_module(&module, Format::Postcard).unwrap();
        let stats = bytecode_stats(&bytecode).unwrap();

        assert!(stats.total_size > 0);
        assert!(stats.header_size > 0);
        assert!(stats.payload_size > 0);
        assert_eq!(stats.format, Format::Postcard);
    }

    #[test]
    fn test_file_roundtrip() {
        let module = create_test_module();
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.zbc");

        serialize_module_to_file(&module, Format::Postcard, &file_path).unwrap();
        let deserialized = deserialize_module_from_file(&file_path).unwrap();

        assert_eq!(module.name, deserialized.name);
        assert_eq!(module.version, deserialized.version);
    }
}
