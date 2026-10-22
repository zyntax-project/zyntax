use std::path::Path;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use zyntax_embed::{Snapshot, SnapshotBuilder, SnapshotCache, SnapshotError};
use zyntax_typed_ast::InternedString;
use zyntax_typed_ast::import_resolver::ModuleArchitecture;

use super::{LoweredProject, ProjectError, SourceProject, canonicalize, load_canonical, resolve};
use crate::{Options, Soundness};

/// Result of an optional persistent frontend cache operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheStatus {
    /// Checked modules were restored without parsing or type checking.
    Hit,
    /// A checked or lowered replacement snapshot was published.
    Stored,
    /// Source was checked, but the cache could not be written.
    Unavailable,
}

/// Check a filesystem project using Zyntax's shared source-validated snapshots.
///
/// All checked sources, including type-only dependencies, and their import
/// resolution are validated before reuse. Unavailable or invalid snapshots fall
/// back to source checking. This operation does no native lowering of its own;
/// it can reuse lowered entries previously prepared by [`load_project_for_runtime`].
pub fn load_project_cached(
    path: impl AsRef<Path>,
    options: Options,
    cache_dir: impl AsRef<Path>,
) -> Result<(SourceProject, CacheStatus), ProjectError> {
    load_cached(path.as_ref(), options, cache_dir.as_ref(), None)
}

/// Install TypeScript primitives and prepare a source-validated project for execution.
/// A cold project is lowered once with this runtime's native policy and published
/// as ordinary split-HIR snapshot data. Hits keep function bodies encoded until
/// the shared runtime reaches them; execution still starts in the interpreter.
pub fn load_project_for_runtime(
    path: impl AsRef<Path>,
    options: Options,
    cache_dir: impl AsRef<Path>,
    runtime: &mut zyntax_embed::TieredRuntime,
) -> Result<(SourceProject, CacheStatus), ProjectError> {
    crate::register_runtime(runtime)?;
    load_cached(path.as_ref(), options, cache_dir.as_ref(), Some(runtime))
}

fn load_cached(
    path: &Path,
    options: Options,
    cache_dir: &Path,
    runtime: Option<&mut zyntax_embed::TieredRuntime>,
) -> Result<(SourceProject, CacheStatus), ProjectError> {
    let path = canonicalize(path)?;
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
    let restored = cache
        .load(|file, specifier| resolve(&architecture, file, specifier).ok())
        .and_then(|snapshot| {
            let lowered = snapshot.carries_hir(identity);
            let sources = if runtime.is_some() && !lowered {
                snapshot.cached_sources()?.to_vec()
            } else {
                vec![]
            };
            Some((restore(snapshot, identity)?, sources, lowered))
        });
    let (project, sources) = if let Some((project, sources, lowered)) = restored {
        if lowered || runtime.is_none() {
            return Ok((project, CacheStatus::Hit));
        }
        (project, sources)
    } else {
        let mut sources = vec![];
        let project = load_canonical(&path, options, Some(&mut sources))?;
        (project, sources)
    };
    if let Some(runtime) = runtime {
        let (program, resolver) = project.into_parts();
        runtime.add_compiled_import_resolver(resolver);
        let (program, hir) = runtime.prepare_typed_program_snapshot(identity, program)?;
        let prepare = || -> Result<Snapshot, SnapshotError> {
            let builder =
                SnapshotBuilder::new("typescript").module_lowered(identity, program, &hir)?;
            cache.prepare(builder, sources)
        };
        let snapshot = prepare().map_err(|error| {
            ProjectError::Native(zyntax_embed::RuntimeError::Execution(error.to_string()))
        })?;
        let status = if cache.publish(&snapshot).is_ok() {
            CacheStatus::Stored
        } else {
            CacheStatus::Unavailable
        };
        let project = restore(snapshot, identity).ok_or_else(|| {
            ProjectError::Native(zyntax_embed::RuntimeError::Execution(
                "prepared project snapshot is unreadable".into(),
            ))
        })?;
        return Ok((project, status));
    }
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
    if snapshot.carries_hir(entry) {
        snapshot.reserve_type_ids();
        let files = snapshot.cached_sources()?.len();
        if snapshot.module_names().count() != 1 {
            return None;
        }
        let import = snapshot.into_modules().ok()?.pop()?;
        let hir = Arc::clone(import.hir()?);
        return Some(SourceProject {
            entry: import.into_program(),
            dependencies: FxHashMap::default(),
            lowered: Some(LoweredProject {
                name: InternedString::new_global(entry),
                hir,
                files,
            }),
        });
    }
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
        lowered: None,
    })
}
