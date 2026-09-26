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

    /// One reservation, handed out in page-aligned segments, one open
    /// segment per kind of piece. Segments are taken from the top of the
    /// range down, the order the system places separate mappings in,
    /// which the instruction TLB fares better with than the reverse.
    ///
    /// Pieces are written while their pages are read-write, and a
    /// finalization gives every segment written since the last one its
    /// final protection and closes it, so pages that were handed out
    /// never change protection again: another thread may be running or
    /// reading them. The next piece of each kind starts a page of its
    /// own.
    pub(super) struct Reserved {
        base: usize,
        /// Bytes reserved, a whole number of pages.
        size: usize,
        page: usize,
        /// Offset of the lowest page a segment has taken.
        bottom: usize,
        open: [Segment; 3],
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
            // `ZYNTAX_TRACE_JIT_MEMORY=1` reports a module's placed bytes
            // each time they pass a power of two MiB; safe to run with.
            let report_at = std::env::var_os("ZYNTAX_TRACE_JIT_MEMORY").map(|_| 1 << 20);
            Ok(Self {
                base: base as usize,
                size,
                page,
                bottom: size,
                open: [Segment::default(); 3],
                unsealed: Vec::new(),
                live: false,
                finalizations: 0,
                report_at,
            })
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
            self.base = 0;
            self.live = false;
        }

        fn finalize(&mut self, branch_protection: BranchProtection) -> ModuleResult<()> {
            let fail = |err| ModuleError::Allocation { err };
            for kind in [CODE, READ_ONLY, WRITABLE] {
                let seg = std::mem::take(&mut self.open[kind]);
                if seg.start != seg.end {
                    self.unsealed.push((seg, kind));
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
            self.live |= self.used() > 0;
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
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Every piece lands inside the range, and after a finalization
        /// each kind starts a page of its own, leaving what was handed
        /// out with the protection it was given.
        #[test]
        fn pieces_stay_in_the_range_and_finalized_pages_stay_sealed() {
            let mut r = Reserved::new(64 << 20).expect("reserved");
            let (base, size, page) = (r.base, r.size, r.page);
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
                        assert_ne!(p / page, last[i] / page, "kind {i}, round {round}");
                    }
                    last[i] = p;
                }
                r.finalize(BranchProtection::None).expect("finalized");
            }
            // SAFETY: placed and finalized above; data stays readable.
            let byte = unsafe { *(last[READ_ONLY] as *const u8) };
            assert_eq!(byte, 3);
        }

        /// A piece past the end of the range is refused with an error.
        #[test]
        fn a_full_range_refuses_the_next_piece() {
            let mut r = Reserved::new(1).expect("reserved");
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
