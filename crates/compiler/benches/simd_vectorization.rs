use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn scalar_add(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..dst.len() {
        dst[i] = a[i] + b[i];
    }
}

fn chunked_add(dst: &mut [f32], a: &[f32], b: &[f32]) {
    let mut i = 0;
    while i + 4 <= dst.len() {
        dst[i] = a[i] + b[i];
        dst[i + 1] = a[i + 1] + b[i + 1];
        dst[i + 2] = a[i + 2] + b[i + 2];
        dst[i + 3] = a[i + 3] + b[i + 3];
        i += 4;
    }

    while i < dst.len() {
        dst[i] = a[i] + b[i];
        i += 1;
    }
}

fn bench_vector_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector-add");
    for size in [1 << 10, 1 << 14, 1 << 18] {
        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];
        let mut dst = vec![0.0f32; size];

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bencher, _| {
            bencher.iter(|| scalar_add(black_box(&mut dst), black_box(&a), black_box(&b)));
        });

        group.bench_with_input(BenchmarkId::new("chunked", size), &size, |bencher, _| {
            bencher.iter(|| chunked_add(black_box(&mut dst), black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

criterion_group!(simd_benches, bench_vector_add);
criterion_main!(simd_benches);
