//! The resident-size reader the memory tests bound growth with.

mod resident;

/// Touching 200 MB makes the reading grow by about that much, so a
/// bound judged against it measures something on this platform.
#[test]
fn a_touched_allocation_is_counted() {
    const SIZE: usize = 200 << 20;
    let before = resident::bytes();
    let mut block = vec![0u8; SIZE];
    // One write per page makes every page resident.
    for page in block.chunks_mut(4096) {
        page[0] = 1;
    }
    let after = resident::bytes();
    std::hint::black_box(&block);
    let grew = after.saturating_sub(before);
    assert!(
        grew >= (SIZE as u64) * 9 / 10,
        "touching {} MB grew the resident size by {} MB ({before} -> {after} bytes)",
        SIZE >> 20,
        grew >> 20
    );
}
