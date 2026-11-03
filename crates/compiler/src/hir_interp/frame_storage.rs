//! Vacant frame buffers. Running frames own their storage outside the pool.

use crate::value::ZyntaxValue;
use std::sync::atomic::AtomicU64;

// Bound idle storage independently of call depth and the largest executed body.
const MAX_BYTES: usize = 1024 * 1024;
const MAX_FRAMES: usize = 32;

#[derive(Default)]
pub(super) struct FrameStorage {
    pub regs: Vec<ZyntaxValue>,
    pub union_blocks: Vec<*mut u8>,
    pub visits: Vec<u32>,
    pub slots: Vec<*const AtomicU64>,
    pub deferred: Vec<Option<(u32, u32)>>,
    pub wanted: Vec<bool>,
    pub call_args: Vec<ZyntaxValue>,
}

impl FrameStorage {
    pub fn prepare(&mut self, registers: usize, unions: usize, sites: usize) {
        debug_assert!(self.regs.is_empty() && self.call_args.is_empty());
        self.regs.resize_with(registers, || ZyntaxValue::Undef);
        self.union_blocks.resize(unions, std::ptr::null_mut());
        self.visits.resize(sites, 0);
        self.slots.resize(sites, std::ptr::null());
        self.deferred.resize(sites, None);
        self.wanted.resize(sites, false);
    }

    fn clear(&mut self) {
        // Drop owned payloads, including arguments left by an error return.
        self.regs.clear();
        self.call_args.clear();
        self.union_blocks.clear();
        self.visits.clear();
        self.slots.clear();
        self.deferred.clear();
        self.wanted.clear();
    }

    fn bytes(&self) -> usize {
        self.regs.capacity() * size_of::<ZyntaxValue>()
            + self.call_args.capacity() * size_of::<ZyntaxValue>()
            + self.union_blocks.capacity() * size_of::<*mut u8>()
            + self.visits.capacity() * size_of::<u32>()
            + self.slots.capacity() * size_of::<*const AtomicU64>()
            + self.deferred.capacity() * size_of::<Option<(u32, u32)>>()
            + self.wanted.capacity() * size_of::<bool>()
    }
}

#[derive(Default)]
pub(super) struct FramePool {
    frames: Vec<FrameStorage>,
    bytes: usize,
}

impl FramePool {
    pub fn take(&mut self) -> FrameStorage {
        let frame = self.frames.pop().unwrap_or_default();
        self.bytes -= frame.bytes();
        frame
    }

    /// Called only after the frame's root guard has been removed.
    pub fn recycle(&mut self, mut frame: FrameStorage) {
        frame.clear();
        let bytes = frame.bytes();
        if self.frames.len() < MAX_FRAMES && bytes <= MAX_BYTES - self.bytes {
            self.bytes += bytes;
            self.frames.push(frame);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_clears_payloads_and_loop_state() {
        let mut pool = FramePool::default();
        let mut frame = pool.take();
        frame.prepare(8, 3, 2);
        frame.regs[0] = ZyntaxValue::String("owned register".into());
        frame
            .call_args
            .push(ZyntaxValue::String("owned argument".into()));
        frame.visits[0] = 42;
        frame.slots[0] = std::ptr::dangling();
        frame.union_blocks[0] = std::ptr::dangling_mut();
        frame.deferred[0] = Some((1, 12));
        frame.wanted[0] = true;
        let buffer = frame.regs.as_ptr();
        pool.recycle(frame);
        let mut frame = pool.take();
        assert!(frame.regs.is_empty() && frame.call_args.is_empty());
        frame.prepare(4, 1, 1);
        assert_eq!(frame.regs.as_ptr(), buffer);
        assert!(frame.regs.iter().all(|v| matches!(v, ZyntaxValue::Undef)));
        assert_eq!(frame.visits, [0]);
        assert_eq!(frame.slots, [std::ptr::null()]);
        assert_eq!(frame.union_blocks, [std::ptr::null_mut()]);
        assert_eq!(frame.deferred, [None]);
        assert_eq!(frame.wanted, [false]);
        assert_eq!(pool.bytes, 0);
    }

    #[test]
    fn oversized_and_deep_frames_do_not_grow_the_idle_pool() {
        let mut pool = FramePool::default();
        let mut oversized = pool.take();
        oversized.prepare(MAX_BYTES / size_of::<ZyntaxValue>() + 1, 0, 0);
        pool.recycle(oversized);
        assert!(pool.frames.is_empty());
        for _ in 0..MAX_FRAMES + 10 {
            let mut frame = FrameStorage::default();
            frame.prepare(16, 0, 0);
            pool.recycle(frame);
        }
        assert_eq!(pool.frames.len(), MAX_FRAMES);
        assert!(pool.bytes <= MAX_BYTES);
        while !pool.frames.is_empty() {
            pool.take();
        }
        assert_eq!(pool.bytes, 0);
    }
}
