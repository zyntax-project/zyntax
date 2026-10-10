//! Where a Cranelift JIT module places its code and data.
//!
//! A module's functions call each other, and reach its data, with
//! PC-relative references, which reach 2 GiB on x86-64. cranelift-jit's
//! default provider maps every piece wherever the system puts it, so
//! pieces of one module can land further apart than that once anything
//! large is mapped in between. Here a module reserves one range below
//! that reach when it is made and places every piece inside it.

use cranelift_jit::JITMemoryProvider;

/// Address space each module reserves: under 2 GiB by a margin wider
/// than any relocation's addend, so every PC-relative reference between
/// two pieces inside it reaches. It costs address space; memory is
/// committed only as pieces are placed.
pub const RESERVATION: usize = (2 << 30) - (1 << 20);

/// The memory provider for a new module: a reservation of `bytes`, or
/// `None`, leaving cranelift-jit's default, when none can be had.
#[cfg(unix)]
pub(crate) fn provider(bytes: usize) -> Option<Box<dyn JITMemoryProvider + Send>> {
    match reserved::Reserved::new(bytes) {
        Ok(reserved) => Some(Box::new(reserved)),
        Err(e) => {
            log::warn!(
                "no {} MiB JIT range could be reserved ({e}); code and data are mapped piece by piece",
                bytes >> 20
            );
            None
        }
    }
}

/// Let this thread write the JIT code of modules whose code lives in a
/// range mapped for JIT code (Apple Silicon), until [`end_code_writes`].
/// Other threads go on running code in the same pages meanwhile.
/// Elsewhere code pages are made writable by protection and this does
/// nothing.
pub(crate) fn begin_code_writes() {
    #[cfg(all(unix, target_os = "macos", target_arch = "aarch64"))]
    reserved::jit_code_writes(true);
}

/// Put this thread back to running JIT code; see [`begin_code_writes`].
/// Safe to call when it is not writing.
pub(crate) fn end_code_writes() {
    #[cfg(all(unix, target_os = "macos", target_arch = "aarch64"))]
    reserved::jit_code_writes(false);
}

/// Windows keeps cranelift-jit's default provider. The one here is built
/// on mmap and mprotect, and cranelift-jit's own arena reserves through
/// `region`, which on Windows commits the whole range at once and so
/// takes its full size from the commit limit.
#[cfg(not(unix))]
pub(crate) fn provider(_bytes: usize) -> Option<Box<dyn JITMemoryProvider + Send>> {
    None
}

#[cfg(unix)]
mod reserved {
    use cranelift_jit::{BranchProtection, JITMemoryKind, JITMemoryProvider};
    use cranelift_module::{ModuleError, ModuleResult};
    use std::io;

    /// Pages of one kind, `[start, end)` from the reservation's base,
    /// filled up to `next`.
    #[derive(Clone, Copy, Default)]
    struct Segment {
        start: usize,
        end: usize,
        next: usize,
    }

    const CODE: usize = 0;
    const READ_ONLY: usize = 1;
    const WRITABLE: usize = 2;

    /// Where code goes when it has a range of its own mapped for JIT
    /// code: placed one piece after another across finalizations, the
    /// compiling thread switched to writing it while others run it.
    struct CodeRange {
        base: usize,
        size: usize,
        next: usize,
        /// Bytes written since the last finalization, `[lo, hi)`.
        dirty: Option<(usize, usize)>,
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    unsafe extern "C" {
        fn pthread_jit_write_protect_np(enabled: libc::c_int);
    }

    /// Whether any module mapped a JIT code range, so the switch below is
    /// worth making.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    static JIT_CODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// Switch this thread between writing and running JIT code ranges.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn jit_code_writes(writing: bool) {
        if JIT_CODE.load(std::sync::atomic::Ordering::Relaxed) {
            // SAFETY: a per-thread switch with no other effect.
            unsafe { pthread_jit_write_protect_np(if writing { 0 } else { 1 }) };
        }
    }

    /// The reservation for data and, right after it, a JIT code range,
    /// both inside one window of `size` bytes so references between them
    /// reach: `(code base, code size, data base)`. The data goes first:
    /// the system places a hinted JIT range where asked, but not a
    /// hinted reservation next to a JIT range. `None` when the two cannot
    /// be mapped that way.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn map_jit_code(size: usize, page: usize) -> Option<(usize, usize, usize)> {
        let code_size = (size / 4).next_multiple_of(page);
        let data_size = size - code_size;
        let unmap = |at: usize, len: usize| {
            // SAFETY: a mapping made below and handed to nothing.
            unsafe { libc::munmap(at as *mut libc::c_void, len) };
        };
        // SAFETY: a fresh anonymous mapping where the system chooses,
        // inaccessible until pieces are placed in it.
        let data = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                data_size,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if data == libc::MAP_FAILED {
            return None;
        }
        let data = data as usize;
        // SAFETY: as above, asked for the addresses right after the data.
        let code = unsafe {
            libc::mmap(
                (data + data_size) as *mut libc::c_void,
                code_size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT,
                -1,
                0,
            )
        };
        if code == libc::MAP_FAILED {
            unmap(data, data_size);
            return None;
        }
        let code = code as usize;
        let lo = code.min(data);
        let hi = (code + code_size).max(data + data_size);
        if hi - lo > size {
            unmap(code, code_size);
            unmap(data, data_size);
            return None;
        }
        JIT_CODE.store(true, std::sync::atomic::Ordering::Relaxed);
        Some((code, code_size, data))
    }

    /// One reservation, handed out in page-aligned segments, one open
    /// segment per kind of piece. Segments are taken from the top of the
    /// range down, the order the system places separate mappings in,
    /// which the instruction TLB fares better with than the reverse.
    ///
    /// Pieces are written while their pages are read-write. A
    /// finalization seals the code segment written since the last one,
    /// and the next code starts a page of its own; a data segment stays
    /// open until a piece no longer fits, and a read-only one is sealed
    /// at the finalization after that. Pages that were sealed never
    /// change protection again: another thread may be running or reading
    /// them.
    pub(super) struct Reserved {
        base: usize,
        /// Bytes reserved, a whole number of pages.
        size: usize,
        page: usize,
        /// Offset of the lowest page a segment has taken.
        bottom: usize,
        open: [Segment; 3],
        /// Code's own range when it has one; otherwise code takes
        /// segments of the reservation like data.
        code: Option<CodeRange>,
        /// Segments written since the last finalization and no longer
        /// open, with their kind.
        unsealed: Vec<(Segment, usize)>,
        /// Whether a finalization handed pieces out, so that code outside
        /// the module may hold their addresses.
        live: bool,
        finalizations: usize,
        /// With `ZYNTAX_TRACE_JIT_MEMORY`, the use at which to report next.
        report_at: Option<usize>,
    }

    fn protect(at: usize, len: usize, prot: libc::c_int) -> io::Result<()> {
        // SAFETY: callers pass whole pages of this module's reservation.
        if unsafe { libc::mprotect(at as *mut libc::c_void, len, prot) } == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    impl Reserved {
        pub(super) fn new(bytes: usize) -> io::Result<Self> {
            // SAFETY: sysconf has no preconditions.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            let size = bytes.next_multiple_of(page);
            // `ZYNTAX_JIT_CODE_PAGES=1` keeps code in protected segments
            // of the reservation instead of a JIT code range; safe to run
            // with.
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            if std::env::var_os("ZYNTAX_JIT_CODE_PAGES").is_none()
                && let Some((code, code_size, data)) = map_jit_code(size, page)
            {
                let mut reserved = Self::over(data, size - code_size, page);
                reserved.code = Some(CodeRange {
                    base: code,
                    size: code_size,
                    next: 0,
                    dirty: None,
                });
                return Ok(reserved);
            }
            Self::paged(size, page)
        }

        /// A reservation whose code takes protected segments like data.
        fn paged(size: usize, page: usize) -> io::Result<Self> {
            // SAFETY: an anonymous mapping where the system chooses,
            // inaccessible until pieces are placed in it.
            let base = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            if base == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            Ok(Self::over(base as usize, size, page))
        }

        /// A provider placing pieces in the reservation at `base`.
        fn over(base: usize, size: usize, page: usize) -> Self {
            // `ZYNTAX_TRACE_JIT_MEMORY=1` reports a module's placed bytes
            // each time they pass a power of two MiB; safe to run with.
            let report_at = std::env::var_os("ZYNTAX_TRACE_JIT_MEMORY").map(|_| 1 << 20);
            Self {
                base,
                size,
                page,
                bottom: size,
                open: [Segment::default(); 3],
                code: None,
                unsealed: Vec::new(),
                live: false,
                finalizations: 0,
                report_at,
            }
        }

        fn full(&self, size: usize) -> io::Error {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!(
                    "no room for {size} more bytes in this JIT module's range of {} KiB",
                    self.size >> 10
                ),
            )
        }

        /// Where a piece of `size` bytes aligned to `align` goes: in the
        /// open segment of `kind`, or in a fresh one below every segment
        /// taken so far when it does not fit there.
        fn place(&mut self, size: usize, align: usize, kind: usize) -> io::Result<usize> {
            let mut seg = self.open[kind];
            let mut at = seg.next.next_multiple_of(align);
            if seg.start == seg.end || at + size > seg.end {
                let need = size.max(1).next_multiple_of(self.page);
                if need > self.bottom {
                    return Err(self.full(size));
                }
                let from = self.bottom - need;
                protect(self.base + from, need, libc::PROT_READ | libc::PROT_WRITE)?;
                // The segment left behind is sealed at the next finalization.
                if seg.start != seg.end {
                    self.unsealed.push((seg, kind));
                }
                seg = Segment {
                    start: from,
                    end: from + need,
                    next: from,
                };
                at = from;
                self.bottom = from;
            }
            seg.next = at + size;
            self.open[kind] = seg;
            Ok(at)
        }

        fn used(&self) -> usize {
            self.size - self.bottom
        }

        /// Give `seg` the final protection of `kind`.
        fn seal(
            &self,
            seg: Segment,
            kind: usize,
            branch_protection: BranchProtection,
        ) -> io::Result<()> {
            let (at, len) = (self.base + seg.start, seg.end - seg.start);
            match kind {
                WRITABLE => return Ok(()),
                READ_ONLY => return protect(at, len, libc::PROT_READ),
                _ => {}
            }
            // SAFETY: the range holds code this module wrote.
            let cleared = unsafe {
                wasmtime_internal_jit_icache_coherence::clear_cache(at as *const libc::c_void, len)
            };
            if cleared.is_err() {
                return Err(io::Error::other(
                    "the instruction cache could not be cleared",
                ));
            }
            protect(at, len, libc::PROT_READ | libc::PROT_EXEC)?;
            #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
            if branch_protection == BranchProtection::BTI
                && std::arch::is_aarch64_feature_detected!("bti")
            {
                // PROT_BTI, which libc does not name.
                protect(at, len, libc::PROT_READ | libc::PROT_EXEC | 0x10)?;
            }
            let _ = branch_protection;
            Ok(())
        }

        fn report(&mut self) {
            let Some(at) = self.report_at else {
                return;
            };
            let used = self.used();
            if used >= at {
                eprintln!(
                    "[jit-memory] {:#x}: {} MiB of {} MiB placed, {} finalizations",
                    self.base,
                    used >> 20,
                    self.size >> 20,
                    self.finalizations
                );
                self.report_at = Some((used + 1).next_power_of_two());
            }
        }
    }

    impl JITMemoryProvider for Reserved {
        fn allocate(
            &mut self,
            size: usize,
            align: u64,
            kind: JITMemoryKind,
        ) -> io::Result<*mut u8> {
            let kind = match kind {
                JITMemoryKind::Executable => CODE,
                JITMemoryKind::ReadOnly => READ_ONLY,
                JITMemoryKind::Writable => WRITABLE,
            };
            let align = usize::try_from(align)
                .ok()
                .filter(|a| a.is_power_of_two() && *a <= self.page)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("a JIT piece cannot be aligned to {align} bytes, past a page"),
                    )
                })?;
            if kind == CODE
                && let Some(code) = &mut self.code
            {
                let at = code.next.next_multiple_of(align);
                if at + size > code.size {
                    return Err(io::Error::new(
                        io::ErrorKind::OutOfMemory,
                        format!(
                            "no room for {size} more bytes of code in this JIT module's range of {} KiB",
                            code.size >> 10
                        ),
                    ));
                }
                code.next = at + size;
                let (lo, hi) = (code.base + at, code.base + at + size);
                code.dirty = Some(code.dirty.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
                super::begin_code_writes();
                return Ok(lo as *mut u8);
            }
            let at = self.place(size, align, kind)?;
            self.report();
            Ok((self.base + at) as *mut u8)
        }

        unsafe fn free_memory(&mut self) {
            if self.base != 0 {
                // SAFETY: the caller guarantees nothing placed here is
                // used again; the reservation is this provider's own.
                unsafe { libc::munmap(self.base as *mut libc::c_void, self.size) };
            }
            if let Some(code) = self.code.take() {
                // SAFETY: as above, for the code range.
                unsafe { libc::munmap(code.base as *mut libc::c_void, code.size) };
            }
            self.base = 0;
            self.live = false;
        }

        fn finalize(&mut self, branch_protection: BranchProtection) -> ModuleResult<()> {
            let fail = |err| ModuleError::Allocation { err };
            // Code must run once finalized, so its segment is sealed and
            // the next piece starts a page of its own. Data segments stay
            // open for the next finalization's pieces: a writable page
            // never changes protection, and a read-only one is sealed
            // once a new segment replaces it, never opened again.
            if let Some(code) = &mut self.code {
                if let Some((lo, hi)) = code.dirty.take() {
                    // SAFETY: the range holds code this module wrote.
                    let cleared = unsafe {
                        wasmtime_internal_jit_icache_coherence::clear_cache(
                            lo as *const libc::c_void,
                            hi - lo,
                        )
                    };
                    if cleared.is_err() {
                        return Err(fail(io::Error::other(
                            "the instruction cache could not be cleared",
                        )));
                    }
                }
                super::end_code_writes();
            } else {
                let seg = std::mem::take(&mut self.open[CODE]);
                if seg.start != seg.end {
                    self.unsealed.push((seg, CODE));
                }
            }
            // A segment that fails to seal stays listed for the next try.
            while let Some(&(seg, kind)) = self.unsealed.last() {
                self.seal(seg, kind, branch_protection).map_err(fail)?;
                self.unsealed.pop();
            }
            if wasmtime_internal_jit_icache_coherence::pipeline_flush_mt().is_err() {
                return Err(fail(io::Error::other(
                    "the other threads' pipelines could not be flushed",
                )));
            }
            self.finalizations += 1;
            self.live |= self.used() > 0 || self.code.as_ref().is_some_and(|c| c.next > 0);
            Ok(())
        }
    }

    impl Drop for Reserved {
        /// With nothing handed out the whole range is unmapped. Otherwise
        /// the pieces stay mapped, as cranelift-jit's own providers leave
        /// them, since callers may still hold their addresses, and only
        /// the pages no segment took are returned.
        fn drop(&mut self) {
            if self.base == 0 {
                return;
            }
            let from = if self.live { self.bottom } else { self.size };
            if from > 0 {
                // SAFETY: `[0, from)` of this provider's reservation holds
                // no piece that was handed out.
                unsafe { libc::munmap(self.base as *mut libc::c_void, from) };
            }
            if let Some(code) = &self.code
                && !self.live
            {
                // SAFETY: nothing in the code range was handed out.
                unsafe { libc::munmap(code.base as *mut libc::c_void, code.size) };
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Every piece lands inside the range; after a finalization code
        /// starts a page of its own while data shares its page with the
        /// pieces before it.
        #[test]
        fn pieces_stay_in_the_range_and_data_shares_pages() {
            // SAFETY: sysconf has no preconditions.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            let mut r = Reserved::paged(64 << 20, page).expect("reserved");
            let (base, size) = (r.base, r.size);
            let kinds = || {
                [
                    JITMemoryKind::Executable,
                    JITMemoryKind::ReadOnly,
                    JITMemoryKind::Writable,
                ]
            };
            let mut last: [usize; 3] = [0; 3];
            for round in 0..4 {
                for (i, kind) in kinds().into_iter().enumerate() {
                    let p = r.allocate(24, 8, kind).expect("placed") as usize;
                    assert!(p >= base && p + 24 <= base + size);
                    // SAFETY: the piece was just placed, read-write.
                    unsafe { std::ptr::write_bytes(p as *mut u8, round as u8, 24) };
                    if round > 0 {
                        if i == CODE {
                            assert_ne!(p / page, last[i] / page, "code, round {round}");
                        } else {
                            assert_eq!(p / page, last[i] / page, "kind {i}, round {round}");
                        }
                    }
                    last[i] = p;
                }
                r.finalize(BranchProtection::None).expect("finalized");
            }
            // SAFETY: placed and finalized above; data stays readable.
            let byte = unsafe { *(last[READ_ONLY] as *const u8) };
            assert_eq!(byte, 3);
        }

        /// Code placed after a finalization shares the page of the code
        /// before it, and both run.
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        #[test]
        fn code_in_a_jit_range_shares_pages_and_runs() {
            let mut r = Reserved::new(64 << 20).expect("reserved");
            assert!(r.code.is_some(), "a JIT code range was mapped");
            // `mov w0, #imm; ret`
            let ret = |imm: u32| [0x5280_0000 | (imm << 5), 0xd65f_03c0u32];
            let mut fns = Vec::new();
            for imm in [42, 7] {
                let p = r
                    .allocate(8, 16, JITMemoryKind::Executable)
                    .expect("placed");
                // SAFETY: the piece was just placed for writing.
                unsafe { std::ptr::copy_nonoverlapping(ret(imm).as_ptr(), p as *mut u32, 2) };
                r.finalize(BranchProtection::None).expect("finalized");
                fns.push(p as usize);
            }
            assert_eq!(fns[0] / r.page, fns[1] / r.page, "one page holds both");
            for (p, want) in fns.into_iter().zip([42, 7]) {
                // SAFETY: finalized code of the signature written above.
                let f: extern "C" fn() -> u32 = unsafe { std::mem::transmute(p) };
                assert_eq!(f(), want);
            }
        }

        /// A piece past the end of the range is refused with an error.
        #[test]
        fn a_full_range_refuses_the_next_piece() {
            // SAFETY: sysconf has no preconditions.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            let mut r = Reserved::paged(page, page).expect("reserved");
            let page = r.page;
            r.allocate(page, 16, JITMemoryKind::Executable)
                .expect("one page fits");
            let refused = r
                .allocate(1, 1, JITMemoryKind::ReadOnly)
                .expect_err("nothing is left");
            assert_eq!(refused.kind(), io::ErrorKind::OutOfMemory);
        }
    }
}
