//! Zyntax strings.
//!
//! A string is a sixteen-byte header followed by its bytes:
//! ```text
//! offset  0  u32 byte_len
//! offset  4  u32 char_len   code points when TEXT, byte_len otherwise
//! offset  8  u32 aux        cached hash (0 = not computed); a builder's capacity
//! offset 12  u32 flags      TEXT | IMMORTAL | BUILDER | INDEXED | BOXED_PREFIX | MAGIC
//! offset 16  bytes[byte_len]
//! ```
//! The data is sixteen-aligned. An INDEXED string carries a character
//! index after its bytes, padded to four: one twelve-byte record per
//! 64-character block `k`, the u32 byte offset of character `64k`, then
//! seven u8 offsets relative to it of characters `64k+8 .. 64k+56`, then
//! one pad byte.
//!
//! A TEXT string holds valid UTF-8. Without TEXT, `char_len ==
//! byte_len` and only byte operations are meaningful.
//!
//! A string is never written after construction, except by the
//! write-once hash store into `aux` of a string that is not a builder,
//! and by appends to a builder its owner holds alone. Every release
//! goes through [`string_free`].

use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

/// String pointer type: points to the header.
///
/// This is equivalent to `ZrtlStringPtr` in the C SDK.
pub type StringPtr = *mut i32;
pub type StringConstPtr = *const i32;

/// Bytes from a string's address to its data.
pub const STRING_HEADER_SIZE: usize = 16;

/// The bytes are UTF-8 and `char_len` counts code points.
pub const TEXT: u32 = 1 << 0;
/// Static or emitted storage: a release is a no-op and the hash is
/// precomputed, so nothing ever writes to it.
pub const IMMORTAL: u32 = 1 << 1;
/// Growing in place: `aux` is the data area's capacity in bytes.
pub const BUILDER: u32 = 1 << 2;
/// A character index follows the bytes.
pub const INDEXED: u32 = 1 << 3;
/// A [`BOXED_PREFIX_SIZE`]-byte box precedes the header and shares its
/// allocation.
pub const BOXED_PREFIX: u32 = 1 << 4;
/// Set in every header.
pub const MAGIC: u32 = 1 << 31;

/// Bytes of the box a [`BOXED_PREFIX`] string carries in front of it.
pub const BOXED_PREFIX_SIZE: usize = 32;

/// A TEXT string at least this long, with a character wider than a
/// byte, is built with a character index.
const INDEX_MIN_BYTES: usize = 64;
/// Characters per index record, and per sub-offset within one.
const BLOCK_CHARS: usize = 64;
const SUB_CHARS: usize = 8;
const RECORD_BYTES: usize = 12;

/// The header, as it lies at a string's address.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StringHeader {
    pub byte_len: u32,
    pub char_len: u32,
    pub aux: u32,
    pub flags: u32,
}

/// FNV-1a over `bytes`.
pub const fn fnv1a_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
        i += 1;
    }
    h
}

/// The hash a header caches: the low half of [`fnv1a_bytes`], never 0.
pub const fn hash32_of(bytes: &[u8]) -> u32 {
    let h = fnv1a_bytes(bytes) as u32;
    if h == 0 { 1 } else { h }
}

#[inline]
const fn align_up(n: usize, to: usize) -> usize {
    (n + to - 1) & !(to - 1)
}

/// Bytes a UTF-8 character occupies, from its first byte.
#[inline]
const fn utf8_width(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead >= 0xF0 {
        4
    } else if lead >= 0xE0 {
        3
    } else {
        2
    }
}

#[inline]
fn is_lead(b: u8) -> bool {
    (b & 0xC0) != 0x80
}

/// The code points of `bytes` when they are UTF-8, or None.
#[inline]
pub fn text_char_len(bytes: &[u8]) -> Option<usize> {
    if bytes.is_ascii() {
        return Some(bytes.len());
    }
    std::str::from_utf8(bytes).ok()?;
    Some(bytes.iter().filter(|b| is_lead(**b)).count())
}

/// Flags and total size of a string of `byte_len` bytes and `char_len`
/// characters, TEXT or not.
#[inline]
pub const fn layout_of(byte_len: usize, char_len: usize, text: bool) -> (u32, usize) {
    let mut flags = MAGIC;
    let mut total = STRING_HEADER_SIZE + byte_len;
    if text {
        flags |= TEXT;
        if char_len < byte_len && byte_len >= INDEX_MIN_BYTES {
            flags |= INDEXED;
            total = align_up(total, 4) + char_len.div_ceil(BLOCK_CHARS) * RECORD_BYTES;
        }
    }
    (flags, total)
}

/// `(char_len, flags, total_size)` of a string of `bytes`, which must be
/// UTF-8 when `text` is set.
pub fn string_layout(bytes: &[u8], text: bool) -> (u32, u32, usize) {
    let char_len = if text {
        debug_assert!(
            std::str::from_utf8(bytes).is_ok(),
            "TEXT bytes are not UTF-8"
        );
        if bytes.is_ascii() {
            bytes.len()
        } else {
            bytes.iter().filter(|b| is_lead(**b)).count()
        }
    } else {
        bytes.len()
    };
    let (flags, total) = layout_of(bytes.len(), char_len, text);
    (char_len as u32, flags, total)
}

/// Write the header. The one place a header is written.
#[inline]
unsafe fn write_header(ptr: *mut u8, byte_len: usize, char_len: u32, flags: u32, aux: u32) {
    (ptr as *mut StringHeader).write_unaligned(StringHeader {
        byte_len: byte_len as u32,
        char_len,
        aux,
        flags: flags | MAGIC,
    });
}

/// Write a string at `ptr`: the header, `bytes` and, when `flags` has
/// INDEXED, the character index. `ptr` must hold the total size
/// [`string_layout`] gave for these arguments. An IMMORTAL string gets
/// its hash here, so nothing writes to it later.
///
/// # Safety
/// `ptr` must be valid for writes of that size.
pub unsafe fn string_init(ptr: *mut u8, bytes: &[u8], char_len: u32, flags: u32) {
    let aux = if flags & IMMORTAL != 0 {
        hash32_of(bytes)
    } else {
        0
    };
    write_header(ptr, bytes.len(), char_len, flags, aux);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(STRING_HEADER_SIZE), bytes.len());
    if flags & INDEXED != 0 {
        build_index(ptr, bytes, char_len as usize);
    }
}

/// The character index of a string whose header and bytes are written.
unsafe fn build_index(ptr: *mut u8, bytes: &[u8], char_len: usize) {
    let records = ptr.add(align_up(STRING_HEADER_SIZE + bytes.len(), 4));
    std::ptr::write_bytes(records, 0, char_len.div_ceil(BLOCK_CHARS) * RECORD_BYTES);
    let mut ci = 0usize;
    let mut block_start = 0usize;
    for (off, &b) in bytes.iter().enumerate() {
        if !is_lead(b) {
            continue;
        }
        let r = ci % BLOCK_CHARS;
        let rec = records.add((ci / BLOCK_CHARS) * RECORD_BYTES);
        if r == 0 {
            (rec as *mut u32).write_unaligned(off as u32);
            block_start = off;
        } else if r % SUB_CHARS == 0 {
            *rec.add(4 + r / SUB_CHARS - 1) = (off - block_start) as u8;
        }
        ci += 1;
    }
}

/// Allocate a string of `byte_len` bytes and `char_len` characters,
/// have `fill` write the bytes into its data, then complete it.
///
/// # Safety
/// `fill` must write exactly `byte_len` bytes, UTF-8 when `text` is set
/// and holding `char_len` characters.
pub unsafe fn string_build(
    byte_len: usize,
    char_len: usize,
    text: bool,
    fill: impl FnOnce(*mut u8),
) -> StringPtr {
    let (flags, total) = layout_of(byte_len, char_len, text);
    let ptr = crate::heap::alloc(total, 16);
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    write_header(ptr, byte_len, char_len as u32, flags, 0);
    fill(ptr.add(STRING_HEADER_SIZE));
    if flags & INDEXED != 0 {
        let bytes = std::slice::from_raw_parts(ptr.add(STRING_HEADER_SIZE), byte_len);
        build_index(ptr, bytes, char_len);
    }
    ptr as StringPtr
}

/// A string of `bytes` with these layout values, from the table when
/// one serves.
fn make(bytes: &[u8], char_len: u32, text: bool) -> StringPtr {
    if let Some(p) = immortal_for(bytes, text) {
        return p;
    }
    let (flags, total) = layout_of(bytes.len(), char_len as usize, text);
    unsafe {
        let ptr = crate::heap::alloc(total, 16);
        if ptr.is_null() {
            return std::ptr::null_mut();
        }
        string_init(ptr, bytes, char_len, flags);
        ptr as StringPtr
    }
}

/// The complete image of a string constant: header, bytes and index,
/// IMMORTAL with its hash precomputed, zero-padded to a multiple of
/// sixteen and to at least thirty-two bytes, so a sixteen-byte read at
/// the data stays inside it. A backend emits it sixteen-aligned and
/// never writes it.
pub fn encode_constant(bytes: &[u8]) -> Vec<u8> {
    let char_len = text_char_len(bytes);
    let (char_len, flags, total) = string_layout(bytes, char_len.is_some());
    let mut image = vec![0u8; align_up(total, 16).max(2 * STRING_HEADER_SIZE)];
    // SAFETY: the image holds `total` bytes.
    unsafe { string_init(image.as_mut_ptr(), bytes, char_len, flags | IMMORTAL) };
    image
}

// ─── the immortal tables ─────────────────────────────────────────────

/// Bytes from one table entry to the next.
pub const IMMORTAL_STRIDE: usize = 32;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Slot([u8; IMMORTAL_STRIDE]);

/// Every one-character string of U+0000..U+00FF as TEXT, every one-byte
/// string without TEXT, and the empty string: static, never released,
/// never written.
#[repr(C, align(16))]
pub struct ImmortalTable {
    char_text: [Slot; 256],
    byte_str: [Slot; 256],
    empty: Slot,
}

const fn slot(bytes: &[u8], text: bool) -> Slot {
    let mut s = [0u8; IMMORTAL_STRIDE];
    let flags = MAGIC | IMMORTAL | if text { TEXT } else { 0 };
    let head = [
        bytes.len() as u32,
        if bytes.is_empty() { 0 } else { 1 },
        hash32_of(bytes),
        flags,
    ];
    let mut w = 0;
    while w < 4 {
        let b = head[w].to_ne_bytes();
        let mut i = 0;
        while i < 4 {
            s[w * 4 + i] = b[i];
            i += 1;
        }
        w += 1;
    }
    let mut i = 0;
    while i < bytes.len() {
        s[STRING_HEADER_SIZE + i] = bytes[i];
        i += 1;
    }
    Slot(s)
}

const fn build_table() -> ImmortalTable {
    let mut t = ImmortalTable {
        char_text: [Slot([0; IMMORTAL_STRIDE]); 256],
        byte_str: [Slot([0; IMMORTAL_STRIDE]); 256],
        empty: slot(&[], true),
    };
    let mut c = 0;
    while c < 256 {
        t.char_text[c] = if c < 0x80 {
            slot(&[c as u8], true)
        } else {
            slot(&[0xC0 | (c >> 6) as u8, 0x80 | (c & 0x3F) as u8], true)
        };
        t.byte_str[c] = slot(&[c as u8], false);
        c += 1;
    }
    t
}

static OWN_TABLE: ImmortalTable = build_table();

/// The table this copy of the SDK hands out: its own, or the host's
/// once [`set_immortal_table`] installed it.
static TABLE: AtomicPtr<ImmortalTable> = AtomicPtr::new(std::ptr::null_mut());

#[inline]
fn table() -> &'static ImmortalTable {
    let p = TABLE.load(Ordering::Relaxed);
    if p.is_null() {
        &OWN_TABLE
    } else {
        // SAFETY: only a host's static table is installed.
        unsafe { &*p }
    }
}

/// The table to hand a plugin loaded from a file, so the process has
/// one set of immortal strings.
pub fn immortal_table() -> *const ImmortalTable {
    table()
}

/// Hand out `t`'s entries in place of this copy's own. Install before
/// the first string is made, as with [`crate::heap::set_allocator`].
///
/// # Safety
/// `t` must be a host's [`immortal_table`], which lives for the process.
pub unsafe fn set_immortal_table(t: *const ImmortalTable) {
    TABLE.store(t as *mut ImmortalTable, Ordering::SeqCst);
}

/// Address of the TEXT entry of U+0000; that of `c` is
/// `IMMORTAL_STRIDE * c` bytes on.
pub fn char_text_base() -> usize {
    table().char_text.as_ptr() as usize
}

/// Address of the one-byte entry of 0; that of `b` is
/// `IMMORTAL_STRIDE * b` bytes on.
pub fn byte_str_base() -> usize {
    table().byte_str.as_ptr() as usize
}

/// The one-character TEXT string of U+0000..U+00FF.
#[inline]
pub fn char_text(c: u8) -> StringPtr {
    &table().char_text[c as usize] as *const Slot as StringPtr
}

/// The one-byte string without TEXT.
#[inline]
pub fn byte_str(b: u8) -> StringPtr {
    &table().byte_str[b as usize] as *const Slot as StringPtr
}

/// The empty string.
#[inline]
pub fn empty_string() -> StringPtr {
    &table().empty as *const Slot as StringPtr
}

/// Whether `p` lies in the immortal table in use, which no release may
/// touch.
#[inline]
pub fn is_immortal_string(p: usize) -> bool {
    let base = table() as *const ImmortalTable as usize;
    p.wrapping_sub(base) < std::mem::size_of::<ImmortalTable>()
}

/// The table entry for `bytes`, when there is one.
#[inline]
fn immortal_for(bytes: &[u8], text: bool) -> Option<StringPtr> {
    match *bytes {
        [] => Some(empty_string()),
        [b] if text => Some(char_text(b)),
        [b] => Some(byte_str(b)),
        [b0 @ 0xC2..=0xC3, b1] if text => Some(char_text(((b0 & 0x1F) << 6) | (b1 & 0x3F))),
        _ => None,
    }
}

// ─── reading ─────────────────────────────────────────────────────────

/// The header of a string.
///
/// # Safety
/// The pointer must be a valid string.
#[inline]
pub unsafe fn string_header(ptr: StringConstPtr) -> StringHeader {
    let h = *(ptr as *const StringHeader);
    debug_assert!(h.flags & MAGIC != 0, "{ptr:p} is not a string");
    h
}

/// The flags of a string.
///
/// # Safety
/// The pointer must be a valid string.
#[inline]
pub unsafe fn string_flags(ptr: StringConstPtr) -> u32 {
    string_header(ptr).flags
}

/// Get string length in bytes from a string pointer
///
/// # Safety
/// The pointer must be null or a valid string.
#[inline]
pub unsafe fn string_length(ptr: StringConstPtr) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    debug_assert!(
        (*(ptr as *const StringHeader)).flags & MAGIC != 0,
        "{ptr:p} is not a string"
    );
    *ptr
}

/// Get pointer to the bytes of a string
///
/// # Safety
/// The pointer must be null or a valid string.
#[inline]
pub unsafe fn string_data(ptr: StringConstPtr) -> *const u8 {
    if ptr.is_null() {
        return std::ptr::null();
    }
    debug_assert!(
        (*(ptr as *const StringHeader)).flags & MAGIC != 0,
        "{ptr:p} is not a string"
    );
    (ptr as *const u8).add(STRING_HEADER_SIZE)
}

/// Get mutable pointer to the bytes of a string
///
/// # Safety
/// The pointer must be null or a valid string its caller may write.
#[inline]
pub unsafe fn string_data_mut(ptr: StringPtr) -> *mut u8 {
    string_data(ptr) as *mut u8
}

/// Bytes of a string's own allocation: header, bytes and index. A
/// builder's is its capacity.
///
/// # Safety
/// The pointer must be a valid string.
pub unsafe fn string_size(ptr: StringConstPtr) -> usize {
    let h = string_header(ptr);
    if h.flags & BUILDER != 0 {
        return STRING_HEADER_SIZE + h.aux as usize;
    }
    let (_, total) = layout_of(
        h.byte_len as usize,
        h.char_len as usize,
        h.flags & TEXT != 0,
    );
    total
}

/// Create a new TEXT string from a Rust &str
///
/// Returns a pointer that must be freed with `string_free`.
pub fn string_new(s: &str) -> StringPtr {
    let bytes = s.as_bytes();
    let char_len = if bytes.is_ascii() {
        bytes.len()
    } else {
        s.chars().count()
    };
    make(bytes, char_len as u32, true)
}

/// A string of `bytes`, TEXT when they are UTF-8.
pub fn string_from_bytes(bytes: &[u8]) -> StringPtr {
    match text_char_len(bytes) {
        Some(n) => make(bytes, n as u32, true),
        None => make(bytes, bytes.len() as u32, false),
    }
}

/// A string of `bytes` without TEXT: a byte string.
pub fn bytes_new(bytes: &[u8]) -> StringPtr {
    make(bytes, bytes.len() as u32, false)
}

/// The empty string.
pub fn string_empty() -> StringPtr {
    empty_string()
}

/// `a` followed by `b`: TEXT when both are, otherwise as the joined
/// bytes decide.
pub fn string_concat(a: StringConstPtr, b: StringConstPtr) -> StringPtr {
    // SAFETY: null or strings the caller holds.
    unsafe {
        let (x, y) = (string_as_bytes(a), string_as_bytes(b));
        let text = |p: StringConstPtr| p.is_null() || string_flags(p) & TEXT != 0;
        if text(a) && text(b) {
            let chars = string_char_count(a) + string_char_count(b);
            if x.len() + y.len() <= 2 {
                return make(&[x, y].concat(), chars as u32, true);
            }
            return string_build(x.len() + y.len(), chars, true, |d| {
                std::ptr::copy_nonoverlapping(x.as_ptr(), d, x.len());
                std::ptr::copy_nonoverlapping(y.as_ptr(), d.add(x.len()), y.len());
            });
        }
        string_from_bytes(&[x, y].concat())
    }
}

/// Bytes `start..end` of a string as a string of its own, TEXT when the
/// source is.
///
/// # Safety
/// The pointer must be null or valid, the range inside it and, for a
/// TEXT source, on character boundaries.
pub unsafe fn string_slice(ptr: StringConstPtr, start: usize, end: usize) -> StringPtr {
    let bytes = &string_as_bytes(ptr)[start..end];
    if ptr.is_null() || string_flags(ptr) & TEXT == 0 {
        return make(bytes, bytes.len() as u32, false);
    }
    let chars = if string_char_count(ptr) == string_length(ptr) as usize {
        bytes.len()
    } else {
        bytes.iter().filter(|b| is_lead(**b)).count()
    };
    make(bytes, chars as u32, true)
}

/// Release a string.
///
/// # Safety
/// The pointer must be null or a string made by this SDK or a backend,
/// not used afterwards.
pub unsafe fn string_free(ptr: StringPtr) {
    if ptr.is_null() {
        return;
    }
    let h = string_header(ptr);
    if h.flags & IMMORTAL != 0 {
        return;
    }
    debug_assert!(
        h.flags & BUILDER == 0 || crate::heap::has_allocator(),
        "a builder string is released only through a host allocator"
    );
    let size = string_size(ptr);
    if h.flags & BOXED_PREFIX != 0 {
        crate::heap::free(
            (ptr as *mut u8).sub(BOXED_PREFIX_SIZE),
            size + BOXED_PREFIX_SIZE,
            16,
        );
    } else {
        crate::heap::free(ptr as *mut u8, size, 16);
    }
}

/// Byte offset of character `index` of `bytes`, counted from the front.
pub fn string_char_offset_slow(bytes: &[u8], index: usize) -> Option<usize> {
    let mut remaining = index;
    for (off, &b) in bytes.iter().enumerate() {
        if is_lead(b) {
            if remaining == 0 {
                return Some(off);
            }
            remaining -= 1;
        }
    }
    (remaining == 0).then_some(bytes.len())
}

/// The index records of an INDEXED string.
#[inline]
unsafe fn records(ptr: StringConstPtr, byte_len: usize) -> *const u8 {
    (ptr as *const u8).add(align_up(STRING_HEADER_SIZE + byte_len, 4))
}

/// Byte offset of character `index`, or of the end when `index` is the
/// character count; None past that.
///
/// # Safety
/// The pointer must be null or a valid string.
pub unsafe fn string_char_offset(ptr: StringConstPtr, index: usize) -> Option<usize> {
    if ptr.is_null() {
        return (index == 0).then_some(0);
    }
    let h = string_header(ptr);
    let (byte_len, char_len) = (h.byte_len as usize, h.char_len as usize);
    if index >= char_len {
        return (index == char_len).then_some(byte_len);
    }
    if char_len == byte_len {
        return Some(index);
    }
    let bytes = string_as_bytes(ptr);
    if h.flags & INDEXED == 0 {
        return string_char_offset_slow(bytes, index);
    }
    let rec = records(ptr, byte_len).add((index / BLOCK_CHARS) * RECORD_BYTES);
    let j = (index % BLOCK_CHARS) / SUB_CHARS;
    let mut off = (rec as *const u32).read_unaligned() as usize;
    if j > 0 {
        off += *rec.add(4 + j - 1) as usize;
    }
    for _ in 0..index % SUB_CHARS {
        off += utf8_width(bytes[off]);
    }
    Some(off)
}

/// Index of the character whose bytes hold byte offset `byte`; the
/// character count at or past the end.
///
/// # Safety
/// The pointer must be null or a valid string.
pub unsafe fn string_char_index_of_byte(ptr: StringConstPtr, byte: usize) -> usize {
    if ptr.is_null() {
        return 0;
    }
    let h = string_header(ptr);
    let (byte_len, char_len) = (h.byte_len as usize, h.char_len as usize);
    if byte >= byte_len {
        return char_len;
    }
    if char_len == byte_len {
        return byte;
    }
    let bytes = string_as_bytes(ptr);
    if h.flags & INDEXED == 0 {
        return bytes[..=byte].iter().filter(|b| is_lead(**b)).count() - 1;
    }
    let recs = records(ptr, byte_len);
    let record_off =
        |k: usize| (recs.add(k * RECORD_BYTES) as *const u32).read_unaligned() as usize;
    // The last block starting at or before `byte`.
    let blocks = char_len.div_ceil(BLOCK_CHARS);
    let (mut lo, mut hi) = (0usize, blocks);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if record_off(mid) <= byte {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let rec = recs.add(lo * RECORD_BYTES);
    let block_start = record_off(lo);
    let in_block = (char_len - lo * BLOCK_CHARS).min(BLOCK_CHARS);
    let mut index = lo * BLOCK_CHARS;
    let mut off = block_start;
    for j in 1..in_block.div_ceil(SUB_CHARS) {
        let sub = block_start + *rec.add(4 + j - 1) as usize;
        if sub > byte {
            break;
        }
        index = lo * BLOCK_CHARS + j * SUB_CHARS;
        off = sub;
    }
    loop {
        let next = off + utf8_width(bytes[off]);
        if next > byte {
            return index;
        }
        off = next;
        index += 1;
    }
}

/// The number of characters in a string.
///
/// # Safety
/// The pointer must be null or a valid string.
pub unsafe fn string_char_count(ptr: StringConstPtr) -> usize {
    if ptr.is_null() {
        return 0;
    }
    string_header(ptr).char_len as usize
}

/// The byte range of the character at position `index`, or None past
/// the end.
///
/// # Safety
/// The pointer must be null or a valid string.
pub unsafe fn string_char_range(
    ptr: StringConstPtr,
    index: usize,
) -> Option<std::ops::Range<usize>> {
    if index >= string_char_count(ptr) {
        return None;
    }
    let start = string_char_offset(ptr, index)?;
    let h = string_header(ptr);
    if h.flags & TEXT == 0 {
        return Some(start..start + 1);
    }
    let bytes = string_as_bytes(ptr);
    Some(start..(start + utf8_width(bytes[start])).min(bytes.len()))
}

/// The hash the header caches, computed and stored on first use.
///
/// # Safety
/// The pointer must be a valid string.
pub unsafe fn string_hash32(ptr: StringConstPtr) -> u32 {
    let h = string_header(ptr);
    if h.flags & BUILDER != 0 {
        return hash32_of(string_as_bytes(ptr));
    }
    // SAFETY: aux is a u32 at offset 8, stored to only through this.
    let aux = &*((ptr as *const u8).add(8) as *const AtomicU32);
    let cached = aux.load(Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    debug_assert!(h.flags & IMMORTAL == 0, "an immortal string has its hash");
    let hash = hash32_of(string_as_bytes(ptr));
    aux.store(hash, Ordering::Relaxed);
    hash
}

/// Copy a string. A table entry is its own copy; any other copy is a
/// plain heap string.
///
/// # Safety
/// The source pointer must be null or valid.
pub unsafe fn string_copy(src: StringConstPtr) -> StringPtr {
    if src.is_null() {
        return string_empty();
    }
    if is_immortal_string(src as usize) {
        return src as StringPtr;
    }
    let h = string_header(src);
    if h.flags & BUILDER != 0 {
        return make(string_as_bytes(src), h.char_len, h.flags & TEXT != 0);
    }
    let total = string_size(src);
    let dst = crate::heap::alloc(total, 16);
    if dst.is_null() {
        return std::ptr::null_mut();
    }
    std::ptr::copy_nonoverlapping(src as *const u8, dst, total);
    let flags = &mut (*(dst as *mut StringHeader)).flags;
    *flags &= !(IMMORTAL | BOXED_PREFIX);
    dst as StringPtr
}

/// Compare two strings for equality
///
/// # Safety
/// Both pointers must be valid or null.
pub unsafe fn string_equals(a: StringConstPtr, b: StringConstPtr) -> bool {
    if a == b {
        return true;
    }
    if a.is_null() || b.is_null() {
        return false;
    }
    string_as_bytes(a) == string_as_bytes(b)
}

/// Get string as a Rust &str: a TEXT string always, another when its
/// bytes happen to be UTF-8.
///
/// # Safety
/// The pointer must be null or valid.
pub unsafe fn string_as_str<'a>(ptr: StringConstPtr) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    let bytes = string_as_bytes(ptr);
    if string_flags(ptr) & TEXT != 0 {
        debug_assert!(
            std::str::from_utf8(bytes).is_ok(),
            "TEXT bytes are not UTF-8"
        );
        // SAFETY: a TEXT string holds UTF-8.
        return Some(std::str::from_utf8_unchecked(bytes));
    }
    std::str::from_utf8(bytes).ok()
}

/// Get string as bytes slice
///
/// # Safety
/// The pointer must be null or valid.
pub unsafe fn string_as_bytes<'a>(ptr: StringConstPtr) -> &'a [u8] {
    if ptr.is_null() {
        return &[];
    }
    let len = string_length(ptr) as usize;
    if len == 0 {
        return &[];
    }
    std::slice::from_raw_parts(string_data(ptr), len)
}

/// Non-owning string view (for SDK convenience)
///
/// This does NOT use the inline format - it's just a reference helper.
#[derive(Debug, Clone, Copy)]
pub struct StringView<'a> {
    pub data: &'a str,
}

impl<'a> StringView<'a> {
    /// Create a view from a string pointer
    ///
    /// # Safety
    /// The pointer must be valid and contain valid UTF-8.
    pub unsafe fn from_ptr(ptr: StringConstPtr) -> Option<Self> {
        string_as_str(ptr).map(|data| Self { data })
    }

    /// Get the length in bytes
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a> AsRef<str> for StringView<'a> {
    fn as_ref(&self) -> &str {
        self.data
    }
}

impl<'a> std::fmt::Display for StringView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data)
    }
}

/// Owned string wrapper that manages memory
pub struct OwnedString {
    ptr: NonNull<i32>,
}

impl OwnedString {
    /// Create from a Rust string
    pub fn new(s: &str) -> Option<Self> {
        let ptr = string_new(s);
        NonNull::new(ptr).map(|ptr| Self { ptr })
    }

    /// Create empty
    pub fn empty() -> Option<Self> {
        let ptr = string_empty();
        NonNull::new(ptr).map(|ptr| Self { ptr })
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> StringConstPtr {
        self.ptr.as_ptr()
    }

    /// Get the mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> StringPtr {
        self.ptr.as_ptr()
    }

    /// Get the length
    pub fn len(&self) -> usize {
        unsafe { string_length(self.as_ptr()) as usize }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get as str
    pub fn as_str(&self) -> Option<&str> {
        unsafe { string_as_str(self.as_ptr()) }
    }

    /// Get as bytes
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { string_as_bytes(self.as_ptr()) }
    }

    /// Release ownership (caller must free)
    pub fn into_raw(self) -> StringPtr {
        let ptr = self.ptr.as_ptr();
        std::mem::forget(self);
        ptr
    }

    /// Take ownership from raw pointer
    ///
    /// # Safety
    /// The pointer must have been created by string_new or similar.
    pub unsafe fn from_raw(ptr: StringPtr) -> Option<Self> {
        NonNull::new(ptr).map(|ptr| Self { ptr })
    }
}

impl Drop for OwnedString {
    fn drop(&mut self) {
        unsafe {
            string_free(self.ptr.as_ptr());
        }
    }
}

impl Clone for OwnedString {
    fn clone(&self) -> Self {
        unsafe {
            let ptr = string_copy(self.as_ptr());
            Self {
                ptr: NonNull::new(ptr).expect("string copy failed"),
            }
        }
    }
}

impl std::fmt::Debug for OwnedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_str() {
            Some(s) => write!(f, "OwnedString({:?})", s),
            None => write!(f, "OwnedString(<invalid UTF-8>)"),
        }
    }
}

impl std::fmt::Display for OwnedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_str() {
            Some(s) => write!(f, "{}", s),
            None => write!(f, "<invalid UTF-8>"),
        }
    }
}

impl PartialEq for OwnedString {
    fn eq(&self, other: &Self) -> bool {
        unsafe { string_equals(self.as_ptr(), other.as_ptr()) }
    }
}

impl Eq for OwnedString {}

impl From<&str> for OwnedString {
    fn from(s: &str) -> Self {
        Self::new(s).expect("string allocation failed")
    }
}

impl From<String> for OwnedString {
    fn from(s: String) -> Self {
        Self::new(&s).expect("string allocation failed")
    }
}

// SAFETY: OwnedString owns its memory exclusively
unsafe impl Send for OwnedString {}
unsafe impl Sync for OwnedString {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_new() {
        let s = OwnedString::new("Hello, ZRTL!").unwrap();
        assert_eq!(s.len(), 12);
        assert_eq!(s.as_str(), Some("Hello, ZRTL!"));
    }

    #[test]
    fn test_string_empty() {
        let s = OwnedString::empty().unwrap();
        assert!(s.is_empty());
        assert_eq!(s.as_str(), Some(""));
    }

    #[test]
    fn test_string_clone() {
        let s1 = OwnedString::from("test string");
        let s2 = s1.clone();

        assert_eq!(s1, s2);
        assert_ne!(s1.as_ptr(), s2.as_ptr()); // Different allocations
    }

    #[test]
    fn test_string_equals() {
        let s1 = OwnedString::from("equal");
        let s2 = OwnedString::from("equal");
        let s3 = OwnedString::from("different");

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_raw_functions() {
        unsafe {
            let ptr = string_new("raw test");
            assert_eq!(string_length(ptr), 8);

            let view = string_as_str(ptr).unwrap();
            assert_eq!(view, "raw test");

            string_free(ptr);
        }
    }

    #[test]
    fn header_layout() {
        unsafe {
            let p = string_new("héllo");
            let h = string_header(p);
            assert_eq!(h.byte_len, 6);
            assert_eq!(h.char_len, 5);
            assert_eq!(h.aux, 0);
            assert_eq!(h.flags, MAGIC | TEXT);
            assert_eq!(string_data(p) as usize % 16, 0);
            assert_eq!(string_size(p), 22);
            assert_eq!(string_hash32(p), hash32_of("héllo".as_bytes()));
            assert_eq!(string_header(p).aux, hash32_of("héllo".as_bytes()));
            string_free(p);

            let b = bytes_new(&[0xff, 0x00, 0x41]);
            let h = string_header(b);
            assert_eq!((h.byte_len, h.char_len, h.flags), (3, 3, MAGIC));
            assert_eq!(string_as_str(b), None);
            string_free(b);

            let t = string_from_bytes(&[0xff, 0xfe]);
            assert_eq!(string_flags(t) & TEXT, 0);
            string_free(t);

            let ascii = string_new(&"a".repeat(100));
            assert_eq!(string_flags(ascii) & INDEXED, 0);
            assert_eq!(string_size(ascii), 116);
            string_free(ascii);
        }
    }

    /// A string of `n` characters cycling through widths 1, 2, 3 and 4.
    fn mixed(n: usize) -> String {
        ['a', 'é', '€', '😀'].iter().cycle().take(n).collect()
    }

    #[test]
    fn index_finds_every_character() {
        for n in [64, 65, 1000] {
            let text = mixed(n);
            let starts: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
            unsafe {
                let p = string_new(&text);
                let h = string_header(p);
                assert_eq!(h.char_len as usize, n);
                assert_ne!(h.flags & INDEXED, 0, "{n} characters");
                assert_eq!(
                    string_size(p),
                    align_up(16 + text.len(), 4) + n.div_ceil(64) * 12
                );
                for (i, &start) in starts.iter().enumerate() {
                    assert_eq!(string_char_offset(p, i), Some(start), "char {i} of {n}");
                    let c = text[start..].chars().next().unwrap();
                    assert_eq!(string_char_range(p, i), Some(start..start + c.len_utf8()));
                }
                assert_eq!(string_char_offset(p, n), Some(text.len()));
                assert_eq!(string_char_offset(p, n + 1), None);
                assert_eq!(string_char_range(p, n), None);
                for byte in 0..text.len() {
                    let expect = starts.partition_point(|&s| s <= byte) - 1;
                    assert_eq!(string_char_index_of_byte(p, byte), expect, "byte {byte}");
                }
                assert_eq!(string_char_index_of_byte(p, text.len()), n);
                // The same answers without the index.
                let short = string_new(&mixed(20));
                for (i, (start, _)) in mixed(20).char_indices().enumerate() {
                    assert_eq!(string_char_offset(short, i), Some(start));
                    assert_eq!(string_char_index_of_byte(short, start), i);
                }
                string_free(short);
                let copy = string_copy(p);
                assert_eq!(string_char_offset(copy, n - 1), Some(starts[n - 1]));
                string_free(copy);
                string_free(p);
            }
        }
    }

    #[test]
    fn immortal_strings() {
        unsafe {
            let a = string_new("a");
            assert_eq!(a, char_text(b'a'));
            assert_eq!(string_new("é"), char_text(0xE9));
            assert_eq!(string_as_str(char_text(0xE9)), Some("é"));
            assert_eq!(string_char_count(char_text(0xE9)), 1);
            assert_eq!(bytes_new(b"a"), byte_str(b'a'));
            assert_eq!(string_flags(byte_str(0x80)) & TEXT, 0);
            assert_eq!(string_from_bytes(&[0x80]), byte_str(0x80));
            assert_eq!(string_new(""), empty_string());
            assert!(is_immortal_string(a as usize));
            assert!(is_immortal_string(byte_str(255) as usize));
            assert!(is_immortal_string(empty_string() as usize));
            assert!(!is_immortal_string(
                empty_string() as usize + IMMORTAL_STRIDE
            ));
            assert_eq!(
                char_text_base() + IMMORTAL_STRIDE * 0x41,
                char_text(0x41) as usize
            );
            assert_eq!(byte_str_base() + IMMORTAL_STRIDE * 7, byte_str(7) as usize);
            assert_eq!(string_hash32(a), hash32_of(b"a"));
            // A release is a no-op and leaves the entry as it was.
            string_free(a);
            string_free(empty_string());
            assert_eq!(string_as_str(char_text(b'a')), Some("a"));
            assert_eq!(string_copy(a), a);
        }
    }

    #[test]
    fn constant_image() {
        let image = encode_constant(b"hi");
        assert_eq!(image.len(), 32);
        let h = unsafe { string_header(image.as_ptr() as StringConstPtr) };
        assert_eq!(h.byte_len, 2);
        assert_eq!(h.char_len, 2);
        assert_eq!(h.aux, hash32_of(b"hi"));
        assert_eq!(h.flags, MAGIC | TEXT | IMMORTAL);
        assert_eq!(&image[16..18], b"hi");
        assert!(image[18..].iter().all(|b| *b == 0));

        assert_eq!(encode_constant(b"").len(), 32);
        assert_eq!(encode_constant(&[b'x'; 16]).len(), 32);
        assert_eq!(encode_constant(&[b'x'; 17]).len(), 48);

        let text = mixed(70);
        let image = encode_constant(text.as_bytes());
        assert_eq!(image.len() % 16, 0);
        let p = image.as_ptr() as StringConstPtr;
        let h = unsafe { string_header(p) };
        assert_eq!(h.flags, MAGIC | TEXT | IMMORTAL | INDEXED);
        assert_eq!(h.char_len, 70);
        let (i, _) = text.char_indices().nth(69).unwrap();
        assert_eq!(unsafe { string_char_offset(p, 69) }, Some(i));
        // A copy of a constant is an ordinary string.
        unsafe {
            let copy = string_copy(p);
            assert_eq!(string_flags(copy), MAGIC | TEXT | INDEXED);
            assert_eq!(string_as_bytes(copy), text.as_bytes());
            string_free(copy);
        }

        let binary = encode_constant(&[0xff]);
        let h = unsafe { string_header(binary.as_ptr() as StringConstPtr) };
        assert_eq!(h.flags, MAGIC | IMMORTAL);
    }

    #[test]
    fn built_strings_match_initialised_ones() {
        let text = mixed(200);
        let (a, b) = text.as_bytes().split_at(101);
        unsafe {
            let built = string_build(text.len(), 200, true, |d| {
                std::ptr::copy_nonoverlapping(a.as_ptr(), d, a.len());
                std::ptr::copy_nonoverlapping(b.as_ptr(), d.add(a.len()), b.len());
            });
            let made = string_new(&text);
            let size = string_size(made);
            assert_eq!(string_size(built), size);
            assert_eq!(
                std::slice::from_raw_parts(built as *const u8, size),
                std::slice::from_raw_parts(made as *const u8, size)
            );
            string_free(built);
            string_free(made);
        }
    }
}
