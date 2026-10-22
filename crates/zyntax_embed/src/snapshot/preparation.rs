use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use super::{SnapshotBuilder, SnapshotCache, SnapshotError, SnapshotSource};
use zyntax_compiler::hir::HirModule;
use zyntax_typed_ast::TypedProgram;

use std::path::{Path, PathBuf};

static ACTIVE: AtomicBool = AtomicBool::new(false);

struct Permit;

impl Drop for Permit {
    fn drop(&mut self) {
        ACTIVE.store(false, Ordering::Release);
    }
}

/// Optional cache optimization. Dropping the handle cancels without waiting;
/// the worker owns its HIR and metadata independently of the executing runtime.
pub struct SnapshotPreparation {
    cancelled: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<bool, SnapshotError>>>,
}

impl SnapshotPreparation {
    /// Finish small graphs inline to avoid worker setup and an unfinished cache.
    /// Returns false without modifying larger graphs, which can move to a worker.
    pub fn optimize_inline(
        identity: &str,
        hir: &mut HirModule,
        error_flag_global: Option<&str>,
    ) -> Result<bool, SnapshotError> {
        let mut instructions = 0;
        for (index, function) in hir
            .functions
            .values()
            .filter(|f| {
                !f.is_external
                    && !f.attributes.optimized
                    && f.signature.type_params.is_empty()
                    && f.signature.const_params.is_empty()
            })
            .enumerate()
        {
            instructions += function
                .blocks
                .values()
                .map(|b| b.instructions.len() + b.phis.len())
                .sum::<usize>();
            if index >= 128 || instructions > 4096 {
                return Ok(false);
            }
        }
        let release = hir.automatic_release;
        super::optimize_snapshot_hir(identity, hir, release, error_flag_global, None)
    }

    pub fn start(
        identity: String,
        program: TypedProgram,
        mut hir: HirModule,
        cache: SnapshotCache,
        sources: Vec<SnapshotSource>,
        error_flag_global: Option<String>,
        resolve: impl Fn(&Path, &str) -> Option<PathBuf> + Send + 'static,
    ) -> Option<Self> {
        if ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return None;
        }
        let permit = Permit;
        let cancelled = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&cancelled);
        let worker = std::thread::Builder::new()
            .name("snapshot-preparation".into())
            .stack_size(8 << 20)
            .spawn(move || {
                let _permit = permit;
                let release = hir.automatic_release;
                if !super::optimize_snapshot_hir(
                    &identity,
                    &mut hir,
                    release,
                    error_flag_global.as_deref(),
                    Some(&stop),
                )? {
                    return Ok(false);
                }
                if stop.load(Ordering::Acquire) {
                    return Ok(false);
                }
                let builder = SnapshotBuilder::new(cache.language())
                    .module_lowered(identity, program, &hir)?;
                if stop.load(Ordering::Acquire) {
                    return Ok(false);
                }
                let snapshot = cache.prepare(builder, sources)?;
                if !cache.is_current(&snapshot, resolve) {
                    return Ok(false);
                }
                cache.publish_unless_cancelled(&snapshot, &stop)
            })
            .ok()?;
        Some(Self {
            cancelled,
            worker: Some(worker),
        })
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_finished(&self) -> bool {
        self.worker.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Explicitly wait for prewarming. Ordinary shutdown should drop the handle.
    /// Returns false if cancelled or the checked sources changed.
    pub fn finish(mut self) -> Result<bool, SnapshotError> {
        self.worker
            .take()
            .unwrap()
            .join()
            .map_err(|_| SnapshotError::Encode("snapshot worker panicked".into()))?
    }
}

impl Drop for SnapshotPreparation {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, mpsc};
    use std::time::{Duration, Instant};
    use zyntax_typed_ast::InternedString;

    static SERIAL: Mutex<()> = Mutex::new(());

    fn candidate(
        root: &Path,
    ) -> (
        String,
        TypedProgram,
        HirModule,
        SnapshotCache,
        Vec<SnapshotSource>,
    ) {
        std::fs::create_dir_all(root).unwrap();
        let file = root.join("entry.test");
        std::fs::write(&file, "entry").unwrap();
        let entry = file.canonicalize().unwrap().to_str().unwrap().to_owned();
        let cache = SnapshotCache::new(root.join("cache"), "test", &entry, "test", "test");
        let sources = vec![SnapshotSource::new(&entry, b"entry", vec![])];
        let hir = HirModule::new(InternedString::new_global(&entry));
        (entry, TypedProgram::default(), hir, cache, sources)
    }

    #[test]
    fn worker_owns_inputs_and_publishes_a_valid_snapshot() {
        let _serial = SERIAL.lock().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let (entry, program, hir, cache, sources) = candidate(directory.path());
        let file = cache.path().to_owned();
        let job = SnapshotPreparation::start(
            entry.clone(),
            program,
            hir,
            cache,
            sources,
            None,
            |_, _| None,
        )
        .unwrap();
        assert!(job.finish().unwrap());
        assert!(!ACTIVE.load(Ordering::Acquire));
        let snapshot = super::super::Snapshot::load_owned(std::fs::read(file).unwrap()).unwrap();
        assert!(snapshot.carries_hir(&entry));
    }

    #[test]
    fn drop_does_not_join_and_busy_workers_are_not_queued() {
        let _serial = SERIAL.lock().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let (entry, program, hir, cache, mut sources) = candidate(directory.path());
        let path = cache.path().to_owned();
        let dependency = directory.path().join("dependency.test");
        std::fs::write(&dependency, "dependency").unwrap();
        let dependency = dependency.canonicalize().unwrap();
        sources[0] = SnapshotSource::new(
            &entry,
            b"entry",
            vec![super::super::SnapshotDependency {
                specifier: "dependency".into(),
                target: dependency.to_str().unwrap().into(),
            }],
        );
        sources.push(SnapshotSource::new(
            dependency.to_str().unwrap(),
            b"dependency",
            vec![],
        ));
        let (ready, waiting) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let job = SnapshotPreparation::start(
            entry.clone(),
            program,
            hir,
            cache,
            sources,
            None,
            move |_, _| {
                ready.send(()).unwrap();
                gate.recv_timeout(Duration::from_secs(10)).unwrap();
                Some(dependency.clone())
            },
        )
        .unwrap();
        waiting.recv_timeout(Duration::from_secs(10)).unwrap();
        let other = candidate(&directory.path().join("other"));
        assert!(
            SnapshotPreparation::start(
                other.0,
                other.1,
                other.2,
                other.3,
                other.4,
                None,
                |_, _| None
            )
            .is_none()
        );
        let (dropped, done) = mpsc::channel();
        let dropper = std::thread::spawn(move || {
            drop(job);
            dropped.send(()).unwrap();
        });
        let returned = done.recv_timeout(Duration::from_secs(2));
        release.send(()).unwrap();
        dropper.join().unwrap();
        assert!(
            returned.is_ok(),
            "drop waited for the worker's blocked resolver"
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        while ACTIVE.load(Ordering::Acquire) {
            assert!(
                Instant::now() < deadline,
                "cancelled worker did not release its permit"
            );
            std::thread::yield_now();
        }
        assert!(!path.exists());
    }

    #[test]
    fn changed_dependencies_are_not_published() {
        let _serial = SERIAL.lock().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let (entry, program, hir, cache, mut sources) = candidate(directory.path());
        let path = cache.path().to_owned();
        let dependency = directory.path().join("dependency.test");
        std::fs::write(&dependency, "before").unwrap();
        let dependency = dependency.canonicalize().unwrap();
        sources[0] = SnapshotSource::new(
            &entry,
            b"entry",
            vec![super::super::SnapshotDependency {
                specifier: "dependency".into(),
                target: dependency.to_str().unwrap().into(),
            }],
        );
        sources.push(SnapshotSource::new(
            dependency.to_str().unwrap(),
            b"before",
            vec![],
        ));
        let (ready, waiting) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let target = dependency.clone();
        let job =
            SnapshotPreparation::start(entry, program, hir, cache, sources, None, move |_, _| {
                ready.send(()).unwrap();
                gate.recv_timeout(Duration::from_secs(10)).unwrap();
                Some(target.clone())
            })
            .unwrap();
        waiting.recv_timeout(Duration::from_secs(10)).unwrap();
        std::fs::write(dependency, "after").unwrap();
        release.send(()).unwrap();
        assert!(!job.finish().unwrap());
        assert!(!path.exists());
    }
}
