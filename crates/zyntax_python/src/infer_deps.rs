//! What one item's inference reads of the module while
//! [`crate::types::infer_module`] runs, and when each part it read last
//! changed: an item none of whose reads changed since it was inferred
//! would be inferred to the same result again, so a round may keep that
//! result instead.

use rustc_hash::FxHashMap as HashMap;
use std::cell::{Cell, RefCell};
use std::hash::BuildHasher;

/// A part of the module that changes while inference runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Dep {
    /// An item's return type, by [`name_key`] of its name.
    Ret(u64),
    /// An item's parameter types, by [`name_key`] of its name. Their
    /// names, number and defaults do not change while inference runs.
    Params(u64),
    /// A class's field of one name: whether the class has it, and its
    /// type. Fields are only added at the end, so no other field's
    /// change moves it.
    Field(usize, u64),
    /// Every field of a class.
    Class(usize),
    /// A closure's signature and whether it escapes.
    Closure(u16),
    /// An item's list facts, by [`name_key`] of its name.
    ListParams(u64),
    /// The kind decided for a class's list field of one name.
    FieldList(usize, u64),
    /// Whether a field name is changed through receivers of no known
    /// class.
    DynamicField(u64),
    /// The per-class types of the fields of one name held by address.
    FieldByClass(u64),
    /// Which dict shapes, and which set shapes, are one class.
    ShapeClasses,
}

/// The key a name's parts go by. Two names sharing a key share their
/// changes, which only makes an item inferred again when it need not be.
pub(crate) fn name_key(name: &str) -> u64 {
    rustc_hash::FxBuildHasher.hash_one(name)
}

/// The key a part is recorded by. Parts sharing a key share their
/// reads and changes, which only makes an item inferred again when it
/// need not be. Never zero, which marks an empty [`Reads::recent`] slot.
fn key_of(dep: Dep) -> u64 {
    rustc_hash::FxBuildHasher.hash_one(dep) | 1
}

#[derive(Default)]
struct Changes {
    /// The change clock: how many changes there have been.
    clock: u64,
    /// The clock at each part's latest change, by its key.
    at: HashMap<u64, u64>,
}

/// What the item being inferred has read.
struct Reads {
    keys: Vec<u64>,
    /// The latest key recorded in each slot, by its top bits: a part
    /// read over and over is recorded once.
    recent: [u64; 128],
}

thread_local! {
    /// Whether changes are counted: while a module is inferred.
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static CHANGES: RefCell<Changes> = RefCell::new(Changes::default());
    /// Whether reads are recorded: while one item is inferred.
    static READING: Cell<bool> = const { Cell::new(false) };
    static READS: RefCell<Reads> = const {
        RefCell::new(Reads {
            keys: Vec::new(),
            recent: [0; 128],
        })
    };
}

/// Changes are counted while this lives, from a clock at zero.
pub(crate) struct Tracking(());

impl Tracking {
    pub(crate) fn start() -> Self {
        CHANGES.with(|c| *c.borrow_mut() = Changes::default());
        TRACKING.with(|t| t.set(true));
        Tracking(())
    }
}

impl Drop for Tracking {
    fn drop(&mut self) {
        TRACKING.with(|t| t.set(false));
        READING.with(|r| r.set(false));
        READS.with(|r| r.borrow_mut().keys.clear());
        CHANGES.with(|c| *c.borrow_mut() = Changes::default());
    }
}

/// The read of `dep`, by the item being inferred if any is.
pub(crate) fn read(dep: Dep) {
    if READING.with(|r| r.get()) {
        let key = key_of(dep);
        READS.with(|r| {
            let mut r = r.borrow_mut();
            let slot = (key >> 57) as usize;
            if r.recent[slot] != key {
                r.recent[slot] = key;
                r.keys.push(key);
            }
        });
    }
}

/// A change to `dep`.
pub(crate) fn wrote(dep: Dep) {
    if TRACKING.with(|t| t.get()) {
        CHANGES.with(|c| {
            let mut c = c.borrow_mut();
            c.clock += 1;
            let now = c.clock;
            c.at.insert(key_of(dep), now);
        });
    }
}

/// The change clock now.
pub(crate) fn now() -> u64 {
    CHANGES.with(|c| c.borrow().clock)
}

/// Record what is read from here to [`finish_reading`].
pub(crate) fn start_reading() {
    READS.with(|r| {
        let mut r = r.borrow_mut();
        r.keys.clear();
        r.recent = [0; 128];
    });
    READING.with(|r| r.set(true));
}

/// The keys of what was read since [`start_reading`], each once.
pub(crate) fn finish_reading() -> Vec<u64> {
    READING.with(|r| r.set(false));
    let mut reads = READS.with(|r| std::mem::take(&mut r.borrow_mut().keys));
    reads.sort_unstable();
    reads.dedup();
    reads
}

/// Whether no part read, by the keys [`finish_reading`] gave, changed
/// after the clock read `since`.
pub(crate) fn unchanged_since(reads: &[u64], since: u64) -> bool {
    CHANGES.with(|c| {
        let c = c.borrow();
        reads
            .iter()
            .all(|k| c.at.get(k).is_none_or(|&at| at <= since))
    })
}
