//! ZPack - Zyntax Package Format
//!
//! A compressed archive format for distributing compiled modules and runtime libraries.
//! ZPack is designed for **JIT execution** - providing dynamic libraries that can be
//! loaded at runtime for symbol resolution.
//!
//! ZPack bundles:
//!
//! - Compiled bytecode modules (.zbc files)
//! - Platform-specific dynamic libraries (.zrtl)
//! - Package metadata (manifest)
//!
//! # Archive Structure
//!
//! ```text
//! my_runtime.zpack (ZIP archive)
//! ├── manifest.json           # Package metadata
//! ├── modules/                 # Compiled bytecode modules
//! │   ├── std/
//! │   │   ├── Array.zbc
//! │   │   ├── String.zbc
//! │   │   └── Math.zbc
//! │   └── main.zbc
//! └── lib/                    # Platform-specific dynamic libraries
//!     ├── x86_64-apple-darwin/
//!     │   └── runtime.zrtl
//!     ├── x86_64-unknown-linux-gnu/
//!     │   └── runtime.zrtl
//!     ├── aarch64-apple-darwin/
//!     │   └── runtime.zrtl
//!     └── x86_64-pc-windows-msvc/
//!         └── runtime.zrtl
//! ```
//!
//! # Compilation Modes
//!
//! - **JIT Mode**: Uses ZPack `.zrtl` (dynamic library) for runtime symbol lookup.
//!   This is the primary use case for ZPack.
//!
//! - **AOT Mode**: Users link static libraries directly at compile time.
//!   ZPack is not used for AOT - instead, pass the `.a` file directly to the linker:
//!   ```bash
//!   zyntax compile --backend llvm source.json -o app
//!   # Then link with runtime:
//!   cc app.o /path/to/libruntime.a -o app
//!   ```
//!
//! # Usage
//!
//! ```bash
//! # JIT execution with runtime from ZPack
//! zyntax compile --jit --pack haxe-runtime.zpack source.json
//! ```
//!
//! # Workflow
//!
//! 1. Compile source files to .zbc bytecode
//! 2. Compile runtime to .zrtl for each target platform
//! 3. Pack everything into .zpack
//!
//! ```bash
//! # Compile source to bytecode
//! zyntax compile --emit-bytecode --output modules/ src/*.hx
//!
//! # Compile runtime for each platform
//! clang -shared -fPIC -o lib/x86_64-apple-darwin/runtime.zrtl runtime.c
//!
//! # Create the zpack
//! zyntax pack --manifest manifest.json --output my_runtime.zpack
//!
//! # Use the zpack
//! zyntax compile --pack my_runtime.zpack --source main.hx
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Seek, Write};
use std::path::Path;
use zip::read::ZipArchive;
use zip::write::{FileOptions, SimpleFileOptions, ZipWriter};
use zip::CompressionMethod;

use crate::bytecode::{deserialize_module, serialize_module, Format};
use crate::zrtl::{ZrtlError, ZrtlPlugin};
use crate::HirModule;

/// File extension for ZPack archives
pub const ZPACK_EXTENSION: &str = "zpack";

/// File extension for bytecode modules
pub const BYTECODE_EXTENSION: &str = "zbc";

/// File extension for dynamic runtime libraries (JIT)
pub const ZRTL_EXTENSION: &str = "zrtl";

/// Current ZPack format version
pub const ZPACK_VERSION: u32 = 1;

/// Default compression method for ZPack archives
///
/// We use DEFLATE (Deflated) for maximum portability - it's supported by
/// all ZIP tools across all platforms. While Zstd offers better compression,
/// DEFLATE is universally compatible.
pub const DEFAULT_COMPRESSION: CompressionMethod = CompressionMethod::Deflated;

/// Compression method for binary files (runtime libraries)
///
/// Binary files are also compressed with DEFLATE for portability.
/// While `Stored` (no compression) would be faster to extract, DEFLATE
/// ensures the zpack file remains portable and reasonably sized.
pub const BINARY_COMPRESSION: CompressionMethod = CompressionMethod::Deflated;

/// Package manifest (manifest.json)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZPackManifest {
    /// Package format version
    pub version: u32,

    /// Package name (e.g., "haxe-std")
    pub name: String,

    /// Package version (semver)
    pub package_version: String,

    /// Package description
    #[serde(default)]
    pub description: String,

    /// Package authors
    #[serde(default)]
    pub authors: Vec<String>,

    /// Package license (SPDX identifier)
    #[serde(default)]
    pub license: Option<String>,

    /// Source language this was compiled from (e.g., "haxe", "zig")
    pub source_language: String,

    /// Entry point module (relative to modules/)
    #[serde(default)]
    pub entry_point: Option<String>,

    /// Dependencies on other zpacks
    #[serde(default)]
    pub dependencies: HashMap<String, String>,

    /// Supported target triples
    #[serde(default)]
    pub targets: Vec<String>,

    /// Module paths in this package
    #[serde(default)]
    pub modules: Vec<String>,

    /// Exported symbols (for documentation/tooling)
    #[serde(default)]
    pub exports: Vec<ZPackExport>,

    /// Custom metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// An exported symbol from the package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZPackExport {
    /// Symbol name (e.g., "$Array$push")
    pub name: String,
    /// Symbol type signature
    #[serde(default)]
    pub signature: Option<String>,
    /// Documentation
    #[serde(default)]
    pub doc: Option<String>,
}

impl Default for ZPackManifest {
    fn default() -> Self {
        Self {
            version: ZPACK_VERSION,
            name: String::new(),
            package_version: "0.1.0".to_string(),
            description: String::new(),
            authors: Vec::new(),
            license: None,
            source_language: "haxe".to_string(),
            entry_point: None,
            dependencies: HashMap::new(),
            targets: Vec::new(),
            modules: Vec::new(),
            exports: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// A compiled bytecode module from the package
#[derive(Debug)]
pub struct ZPackModule {
    /// Module path (e.g., "std/Array")
    pub path: String,
    /// Compiled HIR module
    pub module: HirModule,
}

/// A loaded ZPack
pub struct ZPack {
    /// Package manifest
    pub manifest: ZPackManifest,
    /// Compiled HIR modules (deserialized from bytecode)
    modules: HashMap<String, HirModule>,
    /// Loaded runtime plugin (for current platform, JIT mode)
    pub runtime: Option<ZrtlPlugin>,
    /// Temporary directory for extracted runtime library
    #[allow(dead_code)]
    runtime_temp_dir: Option<tempfile::TempDir>,
}

impl ZPack {
    /// Load a ZPack from a file path
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ZPackError> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| ZPackError::IoError(e.to_string()))?;

        Self::load_from_reader(file)
    }

    /// Load a ZPack from any reader
    pub fn load_from_reader<R: Read + Seek>(reader: R) -> Result<Self, ZPackError> {
        let mut archive =
            ZipArchive::new(reader).map_err(|e| ZPackError::InvalidArchive(e.to_string()))?;

        // Read manifest
        let manifest = Self::read_manifest(&mut archive)?;

        // Read bytecode modules
        let modules = Self::read_modules(&mut archive)?;

        // Extract and load runtime for current platform (JIT mode)
        let (runtime, runtime_temp_dir) = Self::load_runtime(&mut archive)?;

        Ok(Self {
            manifest,
            modules,
            runtime,
            runtime_temp_dir,
        })
    }

    /// Read manifest.json from archive
    fn read_manifest<R: Read + Seek>(
        archive: &mut ZipArchive<R>,
    ) -> Result<ZPackManifest, ZPackError> {
        let mut manifest_file = archive
            .by_name("manifest.json")
            .map_err(|_| ZPackError::MissingManifest)?;

        let mut manifest_json = String::new();
        manifest_file
            .read_to_string(&mut manifest_json)
            .map_err(|e| ZPackError::IoError(e.to_string()))?;

        serde_json::from_str(&manifest_json).map_err(|e| ZPackError::InvalidManifest(e.to_string()))
    }

    /// Read all bytecode modules from archive
    fn read_modules<R: Read + Seek>(
        archive: &mut ZipArchive<R>,
    ) -> Result<HashMap<String, HirModule>, ZPackError> {
        let mut modules = HashMap::new();
        let extension = format!(".{}", BYTECODE_EXTENSION);

        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| ZPackError::InvalidArchive(e.to_string()))?;

            let name = file.name().to_string();

            // Check if it's a bytecode file
            if name.starts_with("modules/") && name.ends_with(&extension) {
                let mut data = Vec::new();
                file.read_to_end(&mut data)
                    .map_err(|e| ZPackError::IoError(e.to_string()))?;

                // Deserialize bytecode to HirModule
                let hir_module = deserialize_module(&data)
                    .map_err(|e| ZPackError::BytecodeError(e.to_string()))?;

                // Convert path: "modules/std/Array.zbc" -> "std/Array"
                let module_path = name
                    .strip_prefix("modules/")
                    .unwrap_or(&name)
                    .strip_suffix(&extension)
                    .unwrap_or(&name)
                    .to_string();

                log::debug!("Loaded module: {}", module_path);
                modules.insert(module_path, hir_module);
            }
        }

        Ok(modules)
    }

    /// Extract and load runtime library for current platform
    fn load_runtime<R: Read + Seek>(
        archive: &mut ZipArchive<R>,
    ) -> Result<(Option<ZrtlPlugin>, Option<tempfile::TempDir>), ZPackError> {
        let target_triple = get_current_target_triple();
        let runtime_path = format!("lib/{}/runtime.zrtl", target_triple);

        // Try to find runtime for current platform
        let has_runtime = (0..archive.len()).any(|i| {
            archive
                .by_index(i)
                .map(|f| f.name().starts_with(&format!("lib/{}/", target_triple)))
                .unwrap_or(false)
        });

        if !has_runtime {
            // Try simplified target
            let simplified = get_simplified_target();
            let alt_path = format!("lib/{}/runtime.zrtl", simplified);

            let has_alt = (0..archive.len()).any(|i| {
                archive
                    .by_index(i)
                    .map(|f| f.name() == alt_path)
                    .unwrap_or(false)
            });

            if !has_alt {
                log::warn!("No runtime found for target '{}' in zpack", target_triple);
                return Ok((None, None));
            }
        }

        // Extract to temp directory
        let temp_dir = tempfile::tempdir().map_err(|e| ZPackError::IoError(e.to_string()))?;

        let zrtl_path = temp_dir.path().join("runtime.zrtl");

        // Try primary target, then simplified
        let actual_path = if archive.by_name(&runtime_path).is_ok() {
            runtime_path
        } else {
            format!("lib/{}/runtime.zrtl", get_simplified_target())
        };

        // Read and write the file
        let mut runtime_file = archive
            .by_name(&actual_path)
            .map_err(|e| ZPackError::InvalidArchive(e.to_string()))?;

        let mut buffer = Vec::new();
        runtime_file
            .read_to_end(&mut buffer)
            .map_err(|e| ZPackError::IoError(e.to_string()))?;

        std::fs::write(&zrtl_path, &buffer).map_err(|e| ZPackError::IoError(e.to_string()))?;

        // Make executable on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&zrtl_path)
                .map_err(|e| ZPackError::IoError(e.to_string()))?
                .permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&zrtl_path, perms)
                .map_err(|e| ZPackError::IoError(e.to_string()))?;
        }

        // Load the plugin
        let plugin =
            ZrtlPlugin::load(&zrtl_path).map_err(|e| ZPackError::RuntimeError(e.to_string()))?;

        log::info!(
            "Loaded runtime '{}' for target '{}'",
            plugin.name(),
            target_triple
        );

        Ok((Some(plugin), Some(temp_dir)))
    }

    /// Get the runtime symbols (if available)
    pub fn runtime_symbols(&self) -> Vec<(&'static str, *const u8)> {
        self.runtime
            .as_ref()
            .map(|r| r.runtime_symbols())
            .unwrap_or_default()
    }

    /// Get runtime symbols with signature information (for auto-boxing)
    pub fn runtime_symbols_with_signatures(&self) -> &[crate::zrtl::RuntimeSymbolInfo] {
        self.runtime
            .as_ref()
            .map(|r| r.symbols_with_signatures())
            .unwrap_or(&[])
    }

    /// Get a module's HIR by path
    pub fn get_module(&self, path: &str) -> Option<&HirModule> {
        self.modules.get(path)
    }

    /// Get all modules
    pub fn modules(&self) -> &HashMap<String, HirModule> {
        &self.modules
    }

    /// List all module paths
    pub fn module_paths(&self) -> Vec<&str> {
        self.modules.keys().map(|s: &String| s.as_str()).collect()
    }

    /// Check if a dynamic runtime is available for the current platform (JIT mode)
    pub fn has_runtime(&self) -> bool {
        self.runtime.is_some()
    }
}

/// ZPack writer for creating archives
pub struct ZPackWriter<W: Write + Seek> {
    writer: ZipWriter<W>,
    manifest: ZPackManifest,
}

impl<W: Write + Seek> ZPackWriter<W> {
    /// Create a new ZPack writer
    pub fn new(writer: W, manifest: ZPackManifest) -> Self {
        Self {
            writer: ZipWriter::new(writer),
            manifest,
        }
    }

    /// Add a compiled HIR module
    pub fn add_module(&mut self, path: &str, hir_module: &HirModule) -> Result<(), ZPackError> {
        let file_path = format!("modules/{}.{}", path, BYTECODE_EXTENSION);

        let options: SimpleFileOptions =
            FileOptions::default().compression_method(DEFAULT_COMPRESSION);

        self.writer
            .start_file(&file_path, options)
            .map_err(|e| ZPackError::IoError(e.to_string()))?;

        // Serialize HirModule to bytecode
        let data = serialize_module(hir_module, Format::Postcard)
            .map_err(|e| ZPackError::BytecodeError(e.to_string()))?;

        self.writer
            .write_all(&data)
            .map_err(|e| ZPackError::IoError(e.to_string()))?;

        // Track module in manifest
        if !self.manifest.modules.contains(&path.to_string()) {
            self.manifest.modules.push(path.to_string());
        }

        Ok(())
    }

    /// Add a pre-serialized bytecode module (raw bytes)
    pub fn add_module_bytes(&mut self, path: &str, data: &[u8]) -> Result<(), ZPackError> {
        let file_path = format!("modules/{}.{}", path, BYTECODE_EXTENSION);

        let options: SimpleFileOptions =
            FileOptions::default().compression_method(DEFAULT_COMPRESSION);

        self.writer
            .start_file(&file_path, options)
            .map_err(|e| ZPackError::IoError(e.to_string()))?;

        self.writer
            .write_all(data)
            .map_err(|e| ZPackError::IoError(e.to_string()))?;

        // Track module in manifest
        if !self.manifest.modules.contains(&path.to_string()) {
            self.manifest.modules.push(path.to_string());
        }

        Ok(())
    }

    /// Add a runtime library for a specific target
    pub fn add_runtime<P: AsRef<Path>>(&mut self, target: &str, path: P) -> Result<(), ZPackError> {
        let file_path = format!("lib/{}/runtime.zrtl", target);

        let options: SimpleFileOptions =
            FileOptions::default().compression_method(BINARY_COMPRESSION); // Don't compress binaries

        self.writer
            .start_file(&file_path, options)
            .map_err(|e| ZPackError::IoError(e.to_string()))?;

        let data = std::fs::read(path.as_ref()).map_err(|e| ZPackError::IoError(e.to_string()))?;

        self.writer
            .write_all(&data)
            .map_err(|e| ZPackError::IoError(e.to_string()))?;

        // Track supported target
        if !self.manifest.targets.contains(&target.to_string()) {
            self.manifest.targets.push(target.to_string());
        }

        Ok(())
    }

    /// Add runtime bytes directly (for cross-compilation)
    pub fn add_runtime_bytes(&mut self, target: &str, data: &[u8]) -> Result<(), ZPackError> {
        let file_path = format!("lib/{}/runtime.zrtl", target);

        let options: SimpleFileOptions =
            FileOptions::default().compression_method(BINARY_COMPRESSION);

        self.writer
            .start_file(&file_path, options)
            .map_err(|e| ZPackError::IoError(e.to_string()))?;

        self.writer
            .write_all(data)
            .map_err(|e| ZPackError::IoError(e.to_string()))?;

        if !self.manifest.targets.contains(&target.to_string()) {
            self.manifest.targets.push(target.to_string());
        }

        Ok(())
    }

    /// Add an export declaration
    pub fn add_export(&mut self, name: &str, signature: Option<&str>, doc: Option<&str>) {
        self.manifest.exports.push(ZPackExport {
            name: name.to_string(),
            signature: signature.map(String::from),
            doc: doc.map(String::from),
        });
    }

    /// Finish writing and close the archive
    pub fn finish(mut self) -> Result<W, ZPackError> {
        // Write manifest
        let manifest_json = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| ZPackError::IoError(e.to_string()))?;

        let options: SimpleFileOptions =
            FileOptions::default().compression_method(DEFAULT_COMPRESSION);

        self.writer
            .start_file("manifest.json", options)
            .map_err(|e| ZPackError::IoError(e.to_string()))?;

        self.writer
            .write_all(manifest_json.as_bytes())
            .map_err(|e| ZPackError::IoError(e.to_string()))?;

        self.writer
            .finish()
            .map_err(|e| ZPackError::IoError(e.to_string()))
    }
}

/// Errors that can occur when working with ZPacks
#[derive(Debug)]
pub enum ZPackError {
    /// I/O error
    IoError(String),
    /// Invalid archive format
    InvalidArchive(String),
    /// Missing manifest.json
    MissingManifest,
    /// Invalid manifest format
    InvalidManifest(String),
    /// Bytecode loading/saving error
    BytecodeError(String),
    /// Runtime loading error
    RuntimeError(String),
    /// Unsupported platform
    UnsupportedPlatform(String),
}

impl std::fmt::Display for ZPackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZPackError::IoError(msg) => write!(f, "I/O error: {}", msg),
            ZPackError::InvalidArchive(msg) => write!(f, "Invalid archive: {}", msg),
            ZPackError::MissingManifest => write!(f, "Missing manifest.json in zpack"),
            ZPackError::InvalidManifest(msg) => write!(f, "Invalid manifest: {}", msg),
            ZPackError::BytecodeError(msg) => write!(f, "Bytecode error: {}", msg),
            ZPackError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            ZPackError::UnsupportedPlatform(triple) => {
                write!(f, "No runtime available for platform: {}", triple)
            }
        }
    }
}

impl std::error::Error for ZPackError {}

impl From<ZrtlError> for ZPackError {
    fn from(err: ZrtlError) -> Self {
        ZPackError::RuntimeError(err.to_string())
    }
}

/// Get the current platform's target triple
pub fn get_current_target_triple() -> &'static str {
    // Use build-time target or runtime detection
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    {
        "x86_64-apple-darwin"
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        "aarch64-apple-darwin"
    }

    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    {
        "x86_64-unknown-linux-gnu"
    }

    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "musl"))]
    {
        "x86_64-unknown-linux-musl"
    }

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        "aarch64-unknown-linux-gnu"
    }

    #[cfg(all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"))]
    {
        "x86_64-pc-windows-msvc"
    }

    #[cfg(all(target_arch = "x86_64", target_os = "windows", target_env = "gnu"))]
    {
        "x86_64-pc-windows-gnu"
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"),
        all(target_arch = "x86_64", target_os = "linux", target_env = "musl"),
        all(target_arch = "aarch64", target_os = "linux"),
        all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"),
        all(target_arch = "x86_64", target_os = "windows", target_env = "gnu"),
    )))]
    {
        "unknown-unknown-unknown"
    }
}

/// Get a simplified target identifier for common platforms
fn get_simplified_target() -> &'static str {
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    {
        "x86_64-apple-darwin"
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        "aarch64-apple-darwin"
    }

    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        "x86_64-unknown-linux-gnu"
    }

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        "aarch64-unknown-linux-gnu"
    }

    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    {
        "x86_64-pc-windows-msvc"
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "linux"),
        all(target_arch = "aarch64", target_os = "linux"),
        all(target_arch = "x86_64", target_os = "windows"),
    )))]
    {
        "unknown"
    }
}

/// Get the dynamic library extension for the current platform
pub fn get_dynamic_lib_extension() -> &'static str {
    ZRTL_EXTENSION
}

/// Common target triples
pub mod targets {
    pub const X86_64_APPLE_DARWIN: &str = "x86_64-apple-darwin";
    pub const AARCH64_APPLE_DARWIN: &str = "aarch64-apple-darwin";
    pub const X86_64_UNKNOWN_LINUX_GNU: &str = "x86_64-unknown-linux-gnu";
    pub const X86_64_UNKNOWN_LINUX_MUSL: &str = "x86_64-unknown-linux-musl";
    pub const AARCH64_UNKNOWN_LINUX_GNU: &str = "aarch64-unknown-linux-gnu";
    pub const X86_64_PC_WINDOWS_MSVC: &str = "x86_64-pc-windows-msvc";
    pub const X86_64_PC_WINDOWS_GNU: &str = "x86_64-pc-windows-gnu";

    /// All commonly supported targets
    pub const ALL: &[&str] = &[
        X86_64_APPLE_DARWIN,
        AARCH64_APPLE_DARWIN,
        X86_64_UNKNOWN_LINUX_GNU,
        X86_64_UNKNOWN_LINUX_MUSL,
        AARCH64_UNKNOWN_LINUX_GNU,
        X86_64_PC_WINDOWS_MSVC,
        X86_64_PC_WINDOWS_GNU,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_serialization() {
        let manifest = ZPackManifest {
            name: "test-pack".to_string(),
            package_version: "1.0.0".to_string(),
            source_language: "haxe".to_string(),
            targets: vec!["x86_64-apple-darwin".to_string()],
            modules: vec!["std/Array".to_string(), "main".to_string()],
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let parsed: ZPackManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, "test-pack");
        assert_eq!(parsed.modules.len(), 2);
    }

    #[test]
    fn test_target_triple_detection() {
        let triple = get_current_target_triple();
        assert!(!triple.is_empty());
        assert!(triple.contains('-'));
    }
}
