use std::path::Path;

use rustc_hash::FxHashMap;
use zyntax_embed::{Snapshot, SnapshotBuilder, SnapshotCache, SnapshotError};
use zyntax_typed_ast::InternedString;
use zyntax_typed_ast::import_resolver::ModuleArchitecture;

use super::{ProjectError, SourceProject, canonicalize, load_canonical, resolve};
use crate::{Options, Soundness};

/// Result of an optional persistent frontend cache operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheStatus {
    /// Checked modules were restored without parsing or type checking.
    Hit,
    /// Source was checked and a replacement snapshot was published.
    Stored,
    /// Source was checked, but the cache could not be written.
    Unavailable,
}

/// Check a filesystem project using Zyntax's shared source-validated snapshots.
///
/// All checked sources, including type-only dependencies, and their import
/// resolution are validated before reuse. Unavailable or invalid snapshots fall
/// back to source checking. Native linking, release analysis and interpreter-first
/// tiering still run normally; these snapshots contain checked AST, not optimized HIR.
pub fn load_project_cached(
    path: impl AsRef<Path>,
    options: Options,
    cache_dir: impl AsRef<Path>,
) -> Result<(SourceProject, CacheStatus), ProjectError> {
    let path = canonicalize(path.as_ref())?;
    let identity = path.to_str().ok_or_else(|| ProjectError::Io {
        file: path.clone(),
        error: std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "module paths must be UTF-8",
        ),
    })?;
    let policy = match options.soundness {
        Soundness::TypeScript => "typescript",
        Soundness::Checked => "checked",
    };
    let cache = SnapshotCache::new(
        cache_dir,
        "typescript",
        identity,
        env!("ZYNTAX_TS_CACHE_BUILD_ID"),
        policy,
    );
    let architecture = ModuleArchitecture::typescript();
    if let Some(project) = cache
        .load(|file, specifier| resolve(&architecture, file, specifier).ok())
        .and_then(|snapshot| restore(snapshot, identity))
    {
        return Ok((project, CacheStatus::Hit));
    }
    let mut sources = vec![];
    let project = load_canonical(&path, options, Some(&mut sources))?;
    let snapshot = (|| -> Result<_, SnapshotError> {
        let mut builder = SnapshotBuilder::new("typescript");
        for source in &sources {
            let program = if source.path() == identity {
                &project.entry
            } else {
                &project.dependencies[&InternedString::new_global(source.path())]
            };
            builder = builder.module_ref(source.path(), program)?;
        }
        cache.store(builder, sources)
    })();
    let status = if snapshot.is_ok() {
        CacheStatus::Stored
    } else {
        CacheStatus::Unavailable
    };
    Ok((project, status))
}

fn restore(snapshot: Snapshot, entry: &str) -> Option<SourceProject> {
    let mut programs = FxHashMap::default();
    for import in snapshot.into_modules().ok()? {
        if import.language() != "typescript" || import.hir().is_some() {
            return None;
        }
        let name = InternedString::new_global(import.module_name());
        programs.insert(name, import.into_program());
    }
    Some(SourceProject {
        entry: programs.remove(&InternedString::new_global(entry))?,
        dependencies: programs,
    })
}
