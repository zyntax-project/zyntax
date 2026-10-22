//! Persistent source validation and atomic publication for ordinary Zyntax snapshots.

#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
use std::path::{Path, PathBuf};

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{SNAPSHOT_EXTENSION, Snapshot};
#[cfg(not(target_arch = "wasm32"))]
use super::{SnapshotBuilder, SnapshotError};

/// An import request and the canonical file it resolved to.
#[derive(Clone, Serialize, Deserialize)]
pub struct SnapshotDependency {
    pub specifier: String,
    pub target: String,
}

/// A checked source file, including dependencies used only during checking.
#[derive(Clone, Serialize, Deserialize)]
pub struct SnapshotSource {
    path: String,
    hash: [u8; 32],
    imports: Vec<SnapshotDependency>,
}

impl SnapshotSource {
    pub fn new(path: impl Into<String>, source: &[u8], imports: Vec<SnapshotDependency>) -> Self {
        Self {
            path: path.into(),
            hash: digest(source),
            imports,
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CacheKey {
    language: String,
    entry: String,
    build: String,
    options: String,
}

#[derive(Serialize, Deserialize)]
pub(super) struct CachedInputs {
    key: CacheKey,
    pub(super) sources: Vec<SnapshotSource>,
    pub(super) blobs_hash: [u8; 32],
}

/// Optional persistence for snapshots of a checked source graph.
///
/// Files use the normal snapshot schema and can be read with [`Snapshot`]. The
/// cache additionally validates source contents, compiler/frontend identity and
/// resolution edges before returning a hit. Cache directories are trusted local
/// build storage, not a boundary for loading untrusted programs.
pub struct SnapshotCache {
    path: PathBuf,
    key: CacheKey,
}

impl SnapshotCache {
    /// `entry` is a canonical UTF-8 filename. `frontend` must fingerprint the parser,
    /// checker, serialized AST dependencies and target. `options` identifies every
    /// setting that affects checking or lowering.
    pub fn new(
        directory: impl AsRef<Path>,
        language: &str,
        entry: &str,
        frontend: &str,
        options: &str,
    ) -> Self {
        let mut hash = Sha256::new();
        for component in [language, entry, options] {
            hash.update((component.len() as u64).to_le_bytes());
            hash.update(component.as_bytes());
        }
        Self {
            path: directory
                .as_ref()
                .join(format!("{:x}.{SNAPSHOT_EXTENSION}", hash.finalize())),
            key: CacheKey {
                language: language.to_owned(),
                entry: entry.to_owned(),
                build: format!(
                    "{frontend}:{}:{}",
                    zyntax_compiler::BUILD_ID,
                    zyntax_compiler::target_pointer_size()
                ),
                options: options.to_owned(),
            },
        }
    }

    pub(super) fn language(&self) -> &str {
        &self.key.language
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read a validated snapshot. Unavailable, stale or corrupt entries are misses.
    /// `resolve` applies the frontend's current module-resolution policy.
    pub fn load(&self, resolve: impl Fn(&Path, &str) -> Option<PathBuf>) -> Option<Snapshot> {
        let snapshot = Snapshot::load_owned(std::fs::read(&self.path).ok()?).ok()?;
        self.is_current(&snapshot, resolve).then_some(snapshot)
    }

    /// Revalidate prepared inputs before a background publication.
    pub fn is_current(
        &self,
        snapshot: &Snapshot,
        resolve: impl Fn(&Path, &str) -> Option<PathBuf>,
    ) -> bool {
        let Some(inputs) = snapshot.cache.as_ref() else {
            return false;
        };
        if inputs.key != self.key
            || snapshot.language() != self.key.language
            || digest(snapshot.blobs()) != inputs.blobs_hash
            || snapshot.validate_compatibility().is_err()
            || !self.validate_sources(&inputs.sources, resolve)
        {
            return false;
        }
        let mut names: FxHashSet<_> = inputs.sources.iter().map(|s| s.path.as_str()).collect();
        // A linked project has one lowered entry and records all of its source
        // dependencies as validation inputs, without duplicating their bodies.
        let linked_project = snapshot.module_names().count() == 1
            && snapshot.carries_hir(&self.key.entry)
            && names.contains(self.key.entry.as_str());
        if !linked_project
            && (snapshot.module_names().count() != names.len()
                || !snapshot.module_names().all(|name| names.remove(name)))
        {
            return false;
        }
        true
    }

    /// Publish a complete snapshot with its checked inputs using an atomic rename.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn store(
        &self,
        builder: SnapshotBuilder,
        sources: Vec<SnapshotSource>,
    ) -> Result<PathBuf, SnapshotError> {
        self.persist(&self.encode(builder, sources)?)
    }

    /// Prepare a source-validated snapshot for immediate use, without publishing it.
    /// The encoded bodies are shared with subsequent readers of this value.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn prepare(
        &self,
        builder: SnapshotBuilder,
        sources: Vec<SnapshotSource>,
    ) -> Result<Snapshot, SnapshotError> {
        Snapshot::load_owned(self.encode(builder, sources)?)
    }

    /// Atomically publish a snapshot prepared by this cache. A publication failure
    /// leaves the prepared value available for execution in the current process.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn publish(&self, snapshot: &Snapshot) -> Result<PathBuf, SnapshotError> {
        if snapshot
            .cache
            .as_ref()
            .is_none_or(|inputs| inputs.key != self.key)
        {
            return Err(SnapshotError::Encode(
                "snapshot belongs to another cache".into(),
            ));
        }
        self.persist(&snapshot.bytes)
    }

    /// Publish only while the preparation remains live. Cancellation is checked
    /// between writes and immediately before the atomic rename.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn publish_unless_cancelled(
        &self,
        snapshot: &Snapshot,
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<bool, SnapshotError> {
        if snapshot
            .cache
            .as_ref()
            .is_none_or(|inputs| inputs.key != self.key)
        {
            return Err(SnapshotError::Encode(
                "snapshot belongs to another cache".into(),
            ));
        }
        self.persist_cancellable(&snapshot.bytes, Some(cancelled))
            .map(|path| path.is_some())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn encode(
        &self,
        mut builder: SnapshotBuilder,
        sources: Vec<SnapshotSource>,
    ) -> Result<Vec<u8>, SnapshotError> {
        if builder.language != self.key.language {
            return Err(SnapshotError::Encode(
                "cache language does not match snapshot".into(),
            ));
        }
        builder.cache = Some(CachedInputs {
            key: self.key.clone(),
            sources,
            blobs_hash: [0; 32],
        });
        builder.encode()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn persist(&self, bytes: &[u8]) -> Result<PathBuf, SnapshotError> {
        Ok(self
            .persist_cancellable(bytes, None)?
            .expect("uncancelled publication"))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn persist_cancellable(
        &self,
        bytes: &[u8],
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Option<PathBuf>, SnapshotError> {
        let stopped =
            || cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire));
        if stopped() {
            return Ok(None);
        }
        let directory = self.path.parent().unwrap();
        let io = |error: std::io::Error| SnapshotError::Encode(error.to_string());
        std::fs::create_dir_all(directory).map_err(io)?;
        let mut file = tempfile::NamedTempFile::new_in(directory).map_err(io)?;
        for chunk in bytes.chunks(64 * 1024) {
            if stopped() {
                return Ok(None);
            }
            file.write_all(chunk).map_err(io)?;
        }
        file.flush().map_err(io)?;
        if stopped() {
            return Ok(None);
        }
        file.persist(&self.path).map_err(|error| io(error.error))?;
        Ok(Some(self.path.clone()))
    }

    fn validate_sources(
        &self,
        sources: &[SnapshotSource],
        resolve: impl Fn(&Path, &str) -> Option<PathBuf>,
    ) -> bool {
        let records: FxHashMap<_, _> = sources.iter().map(|s| (s.path.as_str(), s)).collect();
        if records.len() != sources.len() {
            return false;
        }
        let mut seen = FxHashSet::default();
        let mut pending = vec![self.key.entry.as_str()];
        while let Some(name) = pending.pop() {
            if !seen.insert(name) {
                continue;
            }
            let Some(record) = records.get(name) else {
                return false;
            };
            let path = Path::new(name);
            if path.canonicalize().ok().as_deref() != Some(path) {
                return false;
            }
            let Ok(source) = std::fs::read(path) else {
                return false;
            };
            if digest(&source) != record.hash {
                return false;
            }
            for import in &record.imports {
                if resolve(path, &import.specifier).as_deref() != Some(Path::new(&import.target)) {
                    return false;
                }
                pending.push(&import.target);
            }
        }
        seen.len() == records.len()
    }
}

pub(super) fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn builder(entry: &str) -> SnapshotBuilder {
        SnapshotBuilder::new("test")
            .module_ref(entry, &zyntax_typed_ast::TypedProgram::default())
            .unwrap()
    }

    fn setup(root: &Path) -> (String, SnapshotCache) {
        let source = root.join("source.test");
        std::fs::write(&source, "source").unwrap();
        let entry = source.canonicalize().unwrap().to_str().unwrap().to_owned();
        let cache = SnapshotCache::new(root.join("cache"), "test", &entry, "frontend-1", "strict");
        (entry, cache)
    }

    #[test]
    fn cached_files_are_ordinary_snapshots_with_validated_inputs() {
        let directory = tempfile::tempdir().unwrap();
        let (entry, cache) = setup(directory.path());
        cache
            .store(
                builder(&entry),
                vec![SnapshotSource::new(&entry, b"source", vec![])],
            )
            .unwrap();
        let bytes = std::fs::read(cache.path()).unwrap();
        assert!(bytes.starts_with(super::super::MAGIC));
        assert_eq!(cache.path().extension().unwrap(), SNAPSHOT_EXTENSION);
        assert_eq!(
            Snapshot::load_owned(bytes).unwrap().into_modules().unwrap()[0].module_name(),
            entry
        );
        assert!(cache.load(|_, _| None).is_some());
        for (frontend, options) in [("frontend-2", "strict"), ("frontend-1", "relaxed")] {
            let other = SnapshotCache::new(
                directory.path().join("cache"),
                "test",
                &entry,
                frontend,
                options,
            );
            assert!(other.load(|_, _| None).is_none());
        }
        std::fs::write(&entry, "edited").unwrap();
        assert!(cache.load(|_, _| None).is_none());
    }

    #[test]
    fn a_different_language_cannot_publish_into_the_cache() {
        let directory = tempfile::tempdir().unwrap();
        let (_, cache) = setup(directory.path());
        assert!(cache.store(SnapshotBuilder::new("other"), vec![]).is_err());
        assert!(!cache.path().exists());
    }

    #[test]
    fn prepared_snapshots_can_run_without_publication_and_cannot_cross_cache_keys() {
        let directory = tempfile::tempdir().unwrap();
        let (entry, cache) = setup(directory.path());
        let prepared = cache
            .prepare(
                builder(&entry),
                vec![SnapshotSource::new(&entry, b"source", vec![])],
            )
            .unwrap();
        assert!(!cache.path().exists());
        assert!(prepared.module(&entry).unwrap().is_some());
        let other = SnapshotCache::new(
            directory.path().join("other"),
            "test",
            &entry,
            "different-build",
            "strict",
        );
        assert!(other.publish(&prepared).is_err());
        assert!(!other.path().exists());
        cache.publish(&prepared).unwrap();
        assert!(cache.load(|_, _| None).is_some());
    }

    #[test]
    fn incompatible_lowered_modules_are_cache_misses() {
        let directory = tempfile::tempdir().unwrap();
        let (entry, cache) = setup(directory.path());
        let hir = zyntax_compiler::hir::HirModule::new(
            zyntax_typed_ast::InternedString::new_global(&entry),
        );
        for pointer_mismatch in [false, true] {
            let mut builder = SnapshotBuilder::new("test")
                .module_lowered(&entry, zyntax_typed_ast::TypedProgram::default(), &hir)
                .unwrap();
            let (_, build, pointer) = builder.modules[0].lowered.as_mut().unwrap();
            if pointer_mismatch {
                *pointer = 0;
            } else {
                build.push('x');
            }
            cache
                .store(
                    builder,
                    vec![SnapshotSource::new(&entry, b"source", vec![])],
                )
                .unwrap();
            assert!(cache.load(|_, _| None).is_none());
        }
    }

    #[test]
    fn cancelled_publication_preserves_the_existing_artifact() {
        use std::sync::atomic::AtomicBool;
        let directory = tempfile::tempdir().unwrap();
        let (entry, cache) = setup(directory.path());
        let prepared = cache
            .prepare(
                builder(&entry),
                vec![SnapshotSource::new(&entry, b"source", vec![])],
            )
            .unwrap();
        cache.publish(&prepared).unwrap();
        let original = std::fs::read(cache.path()).unwrap();
        assert!(
            !cache
                .publish_unless_cancelled(&prepared, &AtomicBool::new(true))
                .unwrap()
        );
        assert_eq!(std::fs::read(cache.path()).unwrap(), original);
        assert_eq!(
            std::fs::read_dir(cache.path().parent().unwrap())
                .unwrap()
                .count(),
            1
        );
    }

    #[test]
    fn concurrent_publishers_do_not_expose_partial_snapshots() {
        let directory = tempfile::tempdir().unwrap();
        let (entry, cache) = setup(directory.path());
        cache
            .store(
                builder(&entry),
                vec![SnapshotSource::new(&entry, b"source", vec![])],
            )
            .unwrap();
        let barrier = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            for _ in 0..3 {
                let (barrier, cache, entry) = (&barrier, &cache, &entry);
                scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..12 {
                        cache
                            .store(
                                builder(entry),
                                vec![SnapshotSource::new(entry, b"source", vec![])],
                            )
                            .unwrap();
                    }
                });
            }
            barrier.wait();
            for _ in 0..36 {
                assert!(cache.load(|_, _| None).is_some());
            }
        });
        assert_eq!(
            std::fs::read_dir(cache.path().parent().unwrap())
                .unwrap()
                .count(),
            1
        );
    }
}
