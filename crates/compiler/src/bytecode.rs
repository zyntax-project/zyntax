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

use crate::hir::HirModule;
use std::io::{Read, Write};
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
}

/// Bytecode file header
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BytecodeHeader {
    /// Magic number: "ZBC\0" (0x5A424300)
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
    const MAGIC: u32 = 0x5A424300; // "ZBC\0"
    const CURRENT_MAJOR: u16 = 1;
    const CURRENT_MINOR: u16 = 0;

    /// Create a new header for the given module and format
    pub fn new(module: &HirModule, format: Format) -> Self {
        Self {
            magic: Self::MAGIC,
            major_version: Self::CURRENT_MAJOR,
            minor_version: Self::CURRENT_MINOR,
            format: format.to_u8(),
            flags: 0,
            module_id: *module.id.as_uuid(),
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
        }
    }

    fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Format::Postcard),
            1 => Ok(Format::Json),
            2 => Ok(Format::Bincode),
            _ => Err(BytecodeError::InvalidFormat),
        }
    }
}

// Helper trait to expose UUID from HirId
pub trait AsUuid {
    fn as_uuid(&self) -> &uuid::Uuid;
}

impl AsUuid for crate::hir::HirId {
    fn as_uuid(&self) -> &uuid::Uuid {
        // HirId is a newtype around Uuid, so we can safely transmute
        // This is safe because HirId(Uuid) has the same memory layout as Uuid
        unsafe { &*(self as *const crate::hir::HirId as *const uuid::Uuid) }
    }
}

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
    };

    // Calculate checksum
    let checksum = crc32fast::hash(&payload);

    // Create header
    let mut header = BytecodeHeader::new(module, format);
    header.payload_size = payload.len() as u64;
    header.checksum = checksum;

    // Serialize header using raw 44-byte format (matches deserialize_raw_header)
    let header_bytes = serialize_raw_header(&header);

    // Combine header and payload
    let mut result = Vec::with_capacity(header_bytes.len() + payload.len());
    result.extend_from_slice(&header_bytes);
    result.extend_from_slice(&payload);

    Ok(result)
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
    const HEADER_SIZE: usize = 44;
    if bytes.len() < HEADER_SIZE {
        return Err(BytecodeError::InvalidFormat);
    }

    // Use raw 44-byte header format (matches serialize_raw_header)
    let (header, header_size) = deserialize_raw_header(bytes)?;

    // Validate header
    header.validate()?;

    // Extract payload
    let payload = &bytes[header_size..];

    // Verify checksum
    let checksum = crc32fast::hash(payload);
    if checksum != header.checksum {
        return Err(BytecodeError::ChecksumMismatch);
    }

    // Deserialize payload
    let format = Format::from_u8(header.format)?;
    let module = match format {
        Format::Postcard => postcard::from_bytes(payload)
            .map_err(|e| BytecodeError::DeserializationError(e.to_string()))?,
        Format::Json => serde_json::from_slice(payload)
            .map_err(|e| BytecodeError::DeserializationError(e.to_string()))?,
        Format::Bincode => bincode::deserialize(payload)
            .map_err(|e| BytecodeError::DeserializationError(e.to_string()))?,
    };

    Ok(module)
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
