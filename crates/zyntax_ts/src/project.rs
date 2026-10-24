use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rustc_hash::{FxHashMap, FxHashSet};
use zyntax_embed::{
    CompiledImport, CompiledImportResolverCallback, SnapshotDependency, SnapshotSource,
};
use zyntax_typed_ast::import_resolver::ModuleArchitecture;
use zyntax_typed_ast::{InternedString, TypedDeclaration, TypedProgram};

use crate::{Error, ErrorKind, ModuleDependency, ModuleId, Options, lower};

mod snapshot;
pub use snapshot::{
    CacheStatus, load_project_cached, load_project_for_runtime, load_project_for_runtime_background,
};
pub use zyntax_embed::SnapshotPreparation as CachePreparation;

/// A filesystem failure or a diagnostic in the source file that caused it.
#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    #[error("{0}")]
    Native(#[from] zyntax_embed::RuntimeError),
    #[error("{file}: {error}")]
    Io {
        file: PathBuf,
        error: std::io::Error,
    },
    #[error("{file}: {error}")]
    Diagnostic {
        file: PathBuf,
        source_code: String,
        error: Box<Error>,
    },
}

/// Checked source modules, ready for the shared native linker.
pub struct SourceProject {
    entry: TypedProgram,
    dependencies: FxHashMap<InternedString, TypedProgram>,
    lowered: Option<LoweredProject>,
}

struct LoweredProject {
    name: InternedString,
    hir: Arc<zyntax_compiler::bytecode::LazyModule>,
    files: usize,
}

impl SourceProject {
    pub fn entry(&self) -> &TypedProgram {
        &self.entry
    }

    /// Number of distinct files checked, including dependencies used only as types.
    pub fn module_count(&self) -> usize {
        self.lowered
            .as_ref()
            .map_or(self.dependencies.len() + 1, |lowered| lowered.files)
    }

    /// Move runtime-reachable bodies into a single-use shared import resolver.
    /// Type-only dependencies are dropped before native compilation starts.
    pub fn into_parts(self) -> (TypedProgram, CompiledImportResolverCallback) {
        fn enqueue(program: &TypedProgram, pending: &mut Vec<InternedString>) {
            for declaration in &program.declarations {
                if let TypedDeclaration::Import(import) = &declaration.node {
                    pending.extend(import.module_path.first().copied());
                }
            }
        }
        let Self {
            entry,
            mut dependencies,
            lowered,
        } = self;
        if let Some(LoweredProject { name, hir, .. }) = lowered {
            let language = Some(InternedString::new_global("typescript"));
            let span = Default::default();
            let program = TypedProgram {
                language,
                declarations: vec![zyntax_typed_ast::TypedNode::new(
                    TypedDeclaration::Import(zyntax_typed_ast::TypedImport {
                        language,
                        module_path: vec![name],
                        items: vec![],
                        span,
                    }),
                    zyntax_typed_ast::Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit),
                    span,
                )],
                ..Default::default()
            };
            let name = name.resolve_global().unwrap();
            let mut imports = FxHashMap::default();
            imports.insert(
                name.clone(),
                CompiledImport::new("typescript", name, entry).with_hir(hir),
            );
            return (program, import_resolver(imports));
        }
        let mut pending = vec![];
        enqueue(&entry, &mut pending);
        let mut imports = FxHashMap::default();
        while let Some(name) = pending.pop() {
            if let Some(program) = dependencies.remove(&name) {
                enqueue(&program, &mut pending);
                let name = name.resolve_global().unwrap();
                imports.insert(
                    name.clone(),
                    CompiledImport::new("typescript", name, program),
                );
            }
        }
        (entry, import_resolver(imports))
    }
}

fn import_resolver(imports: FxHashMap<String, CompiledImport>) -> CompiledImportResolverCallback {
    // The callback API is shared and concurrent; this lock is used only while linking.
    let imports = Mutex::new(imports);
    Box::new(move |name: &str| {
        imports
            .lock()
            .map_err(|_| "source import resolver is poisoned".to_owned())
            .map(|mut imports| imports.remove(name))
    })
}

/// Check an entry file and its relative imports using canonical filesystem identities.
/// Each module is parsed once; checking borrows dependency interfaces without cloning bodies.
/// Cycles and dependency chains deeper than 128 files report `Unsupported`.
pub fn load_project(
    path: impl AsRef<Path>,
    options: Options,
) -> Result<SourceProject, ProjectError> {
    let path = canonicalize(path.as_ref())?;
    load_canonical(&path, options, None)
}

fn load_canonical(
    path: &Path,
    options: Options,
    records: Option<&mut Vec<SnapshotSource>>,
) -> Result<SourceProject, ProjectError> {
    let mut loader = Loader {
        options,
        architecture: ModuleArchitecture::typescript(),
        modules: FxHashMap::default(),
        active: FxHashSet::default(),
        stack: vec![],
        records,
    };
    let identity = loader.load(path)?;
    Ok(SourceProject {
        entry: loader.modules.remove(&identity).unwrap(),
        dependencies: loader.modules,
        lowered: None,
    })
}

fn canonicalize(path: &Path) -> Result<PathBuf, ProjectError> {
    path.canonicalize().map_err(|error| ProjectError::Io {
        file: path.to_owned(),
        error,
    })
}

struct Loader<'a> {
    options: Options,
    architecture: ModuleArchitecture,
    modules: FxHashMap<InternedString, TypedProgram>,
    active: FxHashSet<InternedString>,
    stack: Vec<InternedString>,
    records: Option<&'a mut Vec<SnapshotSource>>,
}

impl Loader<'_> {
    fn load(&mut self, file: &Path) -> Result<InternedString, ProjectError> {
        let filename = file.to_str().ok_or_else(|| ProjectError::Io {
            file: file.to_owned(),
            error: std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "module paths must be UTF-8",
            ),
        })?;
        let identity = InternedString::new_global(filename);
        if self.modules.contains_key(&identity) {
            return Ok(identity);
        }
        let module = ModuleId::new(filename);
        let source = std::fs::read_to_string(file).map_err(|error| ProjectError::Io {
            file: file.to_owned(),
            error,
        })?;
        let diagnostic = |error: Error| ProjectError::Diagnostic {
            file: file.to_owned(),
            source_code: source.clone(),
            error: Box::new(error),
        };
        self.active.insert(identity);
        self.stack.push(identity);
        let result = lower::with_parsed(&source, filename, diagnostic, |program, semantic| {
            let mut requests = vec![];
            for statement in &program.body {
                let Some(source) = lower::dependency_source(statement) else {
                    continue;
                };
                // A source or deferred phase import names a module the program does not execute.
                if let oxc_ast::ast::Statement::ImportDeclaration(import) = statement
                    && import.phase.is_some()
                {
                    return Err(diagnostic(Error::new(
                        ErrorKind::Unsupported,
                        "source and deferred import phases are not implemented yet",
                        import.span,
                    )));
                }
                if source.lone_surrogates {
                    return Err(diagnostic(Error::new(
                        ErrorKind::Unsupported,
                        "module specifiers with unpaired surrogates",
                        source.span,
                    )));
                }
                let specifier = source.value.as_str();
                let dependency = self
                    .resolve(file, specifier)
                    .map_err(|error| diagnostic(Error::new(error.0, error.1, source.span)))?;
                let dependency_id = InternedString::new_global(&dependency.to_string_lossy());
                if self.active.contains(&dependency_id) {
                    let mut cycle: Vec<_> = self
                        .stack
                        .iter()
                        .skip_while(|name| **name != dependency_id)
                        .map(|name| name.resolve_global().unwrap())
                        .collect();
                    cycle.push(dependency.to_string_lossy().into_owned());
                    return Err(diagnostic(Error::new(
                        ErrorKind::Unsupported,
                        format!("cyclic module initialization: {}", cycle.join(" -> ")),
                        source.span,
                    )));
                }
                // Retained parser arenas are bounded along the active dependency chain.
                if self.stack.len() >= 128 && !self.modules.contains_key(&dependency_id) {
                    return Err(diagnostic(Error::new(
                        ErrorKind::Unsupported,
                        "module dependency depth exceeds 128",
                        source.span,
                    )));
                }
                self.load(&dependency)?;
                requests.push((specifier, dependency_id));
            }
            let dependencies: Vec<_> = requests
                .iter()
                .map(|(specifier, id)| ModuleDependency {
                    specifier,
                    program: &self.modules[id],
                })
                .collect();
            let program =
                lower::lower_parsed(program, semantic, module, self.options, &dependencies)
                    .map_err(diagnostic)?;
            if let Some(records) = self.records.as_mut() {
                records.push(SnapshotSource::new(
                    filename,
                    source.as_bytes(),
                    requests
                        .iter()
                        .map(|(specifier, target)| SnapshotDependency {
                            specifier: (*specifier).to_owned(),
                            target: target.resolve_global().unwrap(),
                        })
                        .collect(),
                ));
            }
            Ok(program)
        });
        self.stack.pop();
        self.active.remove(&identity);
        let mut program = result?;
        program
            .source_files
            .push(zyntax_typed_ast::source::SourceFile::new(
                filename.to_owned(),
                source,
            ));
        self.modules.insert(identity, program);
        Ok(identity)
    }

    fn resolve(&self, importer: &Path, specifier: &str) -> Result<PathBuf, (ErrorKind, String)> {
        resolve(&self.architecture, importer, specifier)
    }
}

fn resolve(
    architecture: &ModuleArchitecture,
    importer: &Path,
    specifier: &str,
) -> Result<PathBuf, (ErrorKind, String)> {
    if !(specifier.starts_with("./")
        || specifier.starts_with("../")
        || matches!(specifier, "." | ".."))
    {
        return Err((
            ErrorKind::Unsupported,
            format!("package and absolute module resolution: `{specifier}`"),
        ));
    }
    if matches!(
        Path::new(specifier).extension().and_then(|s| s.to_str()),
        Some("mts" | "cts" | "mjs" | "cjs" | "json")
    ) {
        return Err((
            ErrorKind::Unsupported,
            format!("native imports require executable .ts sources: `{specifier}`"),
        ));
    }
    let base = importer.parent().unwrap().to_owned();
    for candidate in architecture.module_to_paths(&[specifier.to_owned()], &base) {
        match std::fs::metadata(&candidate) {
            Ok(metadata) if metadata.is_file() => {
                if candidate.extension().is_none_or(|s| s != "ts")
                    || candidate.to_string_lossy().ends_with(".d.ts")
                {
                    return Err((
                        ErrorKind::Unsupported,
                        format!(
                            "native imports require executable .ts sources: {}",
                            candidate.display()
                        ),
                    ));
                }
                let path = candidate.canonicalize().map_err(|error| {
                    (ErrorKind::Type, format!("{}: {error}", candidate.display()))
                })?;
                if path.to_str().is_none() {
                    return Err((ErrorKind::Unsupported, "module paths must be UTF-8".into()));
                }
                return Ok(path);
            }
            Ok(_) => {}
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                ) => {}
            Err(error) => {
                return Err((ErrorKind::Type, format!("{}: {error}", candidate.display())));
            }
        }
    }
    Err((
        ErrorKind::Type,
        format!("cannot resolve module `{specifier}`"),
    ))
}
