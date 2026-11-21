//! Collection timing over a large linked graph, on whichever heap the
//! environment selects. Run with `--ignored --nocapture`, once with
//! `ZYNTAX_HOST_HEAP=immix` and once without. `ONLY` keeps the shapes
//! whose label contains it, `FAR` makes one nearby link in that many a
//! distant one, and `LARGE` adds that many 64 KB blocks to the roots.

#![cfg(not(target_arch = "wasm32"))]

use zyntax_compiler::collector;
use zyntax_compiler::pool_alloc::zyntax_alloc;

fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

#[test]
#[ignore]
fn mark_speed() {
    zyntax_compiler::host_heap::install_from_env();
    collector::enable();
    collector::set_automatic(false);
    let far: usize = std::env::var("FAR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let only = std::env::var("ONLY").ok();
    let larges: usize = std::env::var("LARGE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    for (label, words, fanout, nobj, local) in [
        ("4w/2p shuffled", 4usize, 2usize, 1_500_000usize, false),
        ("8w/3p shuffled", 8, 3, 800_000, false),
        ("4w/2p local", 4, 2, 1_500_000, true),
        ("8w/3p local", 8, 3, 800_000, true),
        ("2w/1p local", 2, 1, 3_000_000, true),
    ] {
        if only.as_deref().is_some_and(|o| !label.contains(o)) {
            continue;
        }
        let mut objs: Vec<usize> = (0..nobj)
            // SAFETY: a pool allocation of `words` words.
            .map(|_| unsafe { zyntax_alloc(words * 8) as usize })
            .collect();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        if !local {
            for i in (1..objs.len()).rev() {
                let j = rng(&mut seed) as usize % (i + 1);
                objs.swap(i, j);
            }
        }
        for i in 0..objs.len() {
            let o = objs[i] as *mut usize;
            // SAFETY: the object is `words` words long.
            unsafe {
                *o = objs[(i + 1) % nobj];
                for k in 1..words {
                    *o.add(k) = if k <= fanout {
                        let r = rng(&mut seed) as usize;
                        if local && r % far != 0 {
                            objs[(i + (r >> 8) % 256) % nobj]
                        } else {
                            objs[(r >> 8) % nobj]
                        }
                    } else {
                        (rng(&mut seed) % 1000) as usize
                    };
                }
            }
        }
        let mut held = vec![objs[0]];
        for _ in 0..larges {
            // SAFETY: a block of 64 KB, zeroed before it is read.
            unsafe {
                let b = zyntax_alloc(64 << 10);
                std::ptr::write_bytes(b, 0, 64 << 10);
                held.push(b as usize);
            }
        }
        let roots = Box::leak(held.into_boxed_slice());
        let root_addr = roots.as_ptr() as *const u8;
        collector::add_root_range(root_addr, roots.len() * 8);
        let mut times = Vec::new();
        for _ in 0..7 {
            let t = std::time::Instant::now();
            collector::collect();
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "{label}: {} MB live, collect median {:.1} ms, min {:.1} ms",
            collector::stats().live >> 20,
            times[3],
            times[0]
        );
        collector::remove_root_range(root_addr);
        roots.fill(0);
        // Reclaim before the next shape.
        collector::collect();
    }
}
