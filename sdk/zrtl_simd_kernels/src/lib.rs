//! Vector kernels shared by the SIMD plugin and the plugins that build on
//! it.
//!
//! These live outside `plugins/` on purpose. A plugin crate exports the
//! registration statics `_zrtl_info` and `_zrtl_symbols` under fixed names,
//! so linking one plugin into another defines them twice — which the macOS
//! linker tolerates and `lld` rejects. Sharing code through a plain library
//! keeps every plugin a leaf.

use wide::*;

/// Dot product of two f32 vectors
/// Returns a.dot(b) = sum(a[i] * b[i])
#[no_mangle]
pub extern "C" fn vec_dot_product_f32(a: *const f32, b: *const f32, len: u64) -> f32 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }

    let len = len as usize;

    unsafe {
        // Process 8 elements at a time with SIMD
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = f32x8::ZERO;

        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::new([
                *a.add(offset),
                *a.add(offset + 1),
                *a.add(offset + 2),
                *a.add(offset + 3),
                *a.add(offset + 4),
                *a.add(offset + 5),
                *a.add(offset + 6),
                *a.add(offset + 7),
            ]);
            let vb = f32x8::new([
                *b.add(offset),
                *b.add(offset + 1),
                *b.add(offset + 2),
                *b.add(offset + 3),
                *b.add(offset + 4),
                *b.add(offset + 5),
                *b.add(offset + 6),
                *b.add(offset + 7),
            ]);
            sum += va * vb;
        }

        // Horizontal sum of SIMD vector
        let arr = sum.to_array();
        let mut result = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

        // Handle remainder with scalar
        let base = chunks * 8;
        for i in 0..remainder {
            result += *a.add(base + i) * *b.add(base + i);
        }

        result
    }
}

/// Sum all elements of an f32 vector
#[no_mangle]
pub extern "C" fn vec_sum_f32(data: *const f32, len: u64) -> f32 {
    if data.is_null() || len == 0 {
        return 0.0;
    }

    let len = len as usize;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = f32x8::ZERO;

        for i in 0..chunks {
            let offset = i * 8;
            let v = f32x8::new([
                *data.add(offset),
                *data.add(offset + 1),
                *data.add(offset + 2),
                *data.add(offset + 3),
                *data.add(offset + 4),
                *data.add(offset + 5),
                *data.add(offset + 6),
                *data.add(offset + 7),
            ]);
            sum += v;
        }

        let arr = sum.to_array();
        let mut result = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

        let base = chunks * 8;
        for i in 0..remainder {
            result += *data.add(base + i);
        }

        result
    }
}

/// Scale all elements by a scalar: data[i] *= scalar
#[no_mangle]
pub extern "C" fn vec_scale_f32(data: *mut f32, scalar: f32, len: u64) {
    if data.is_null() || len == 0 {
        return;
    }

    let len = len as usize;
    let scalar_vec = f32x8::splat(scalar);

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let v = f32x8::new([
                *data.add(offset),
                *data.add(offset + 1),
                *data.add(offset + 2),
                *data.add(offset + 3),
                *data.add(offset + 4),
                *data.add(offset + 5),
                *data.add(offset + 6),
                *data.add(offset + 7),
            ]);
            let result = v * scalar_vec;
            let arr = result.to_array();
            for j in 0..8 {
                *data.add(offset + j) = arr[j];
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            *data.add(base + i) *= scalar;
        }
    }
}

/// Find minimum value in vector
#[no_mangle]
pub extern "C" fn vec_min_f32(data: *const f32, len: u64) -> f32 {
    if data.is_null() || len == 0 {
        return f32::INFINITY;
    }

    let len = len as usize;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        let mut min_vec = f32x8::splat(f32::INFINITY);

        for i in 0..chunks {
            let offset = i * 8;
            let v = f32x8::new([
                *data.add(offset), *data.add(offset + 1), *data.add(offset + 2), *data.add(offset + 3),
                *data.add(offset + 4), *data.add(offset + 5), *data.add(offset + 6), *data.add(offset + 7),
            ]);
            min_vec = min_vec.min(v);
        }

        let arr = min_vec.to_array();
        let mut result = arr[0].min(arr[1]).min(arr[2]).min(arr[3])
            .min(arr[4]).min(arr[5]).min(arr[6]).min(arr[7]);

        let base = chunks * 8;
        for i in 0..remainder {
            result = result.min(*data.add(base + i));
        }

        result
    }
}

/// Find maximum value in vector
#[no_mangle]
pub extern "C" fn vec_max_f32(data: *const f32, len: u64) -> f32 {
    if data.is_null() || len == 0 {
        return f32::NEG_INFINITY;
    }

    let len = len as usize;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        let mut max_vec = f32x8::splat(f32::NEG_INFINITY);

        for i in 0..chunks {
            let offset = i * 8;
            let v = f32x8::new([
                *data.add(offset), *data.add(offset + 1), *data.add(offset + 2), *data.add(offset + 3),
                *data.add(offset + 4), *data.add(offset + 5), *data.add(offset + 6), *data.add(offset + 7),
            ]);
            max_vec = max_vec.max(v);
        }

        let arr = max_vec.to_array();
        let mut result = arr[0].max(arr[1]).max(arr[2]).max(arr[3])
            .max(arr[4]).max(arr[5]).max(arr[6]).max(arr[7]);

        let base = chunks * 8;
        for i in 0..remainder {
            result = result.max(*data.add(base + i));
        }

        result
    }
}

/// Fill array with constant value: data[i] = value
#[no_mangle]
pub extern "C" fn vec_fill_f32(data: *mut f32, value: f32, len: u64) {
    if data.is_null() || len == 0 {
        return;
    }

    let len = len as usize;
    let value_vec = f32x8::splat(value);

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let arr = value_vec.to_array();
            for j in 0..8 {
                *data.add(offset + j) = arr[j];
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            *data.add(base + i) = value;
        }
    }
}

/// Euclidean distance: sqrt(sum((a[i] - b[i])^2))
#[no_mangle]
pub extern "C" fn vec_euclidean_f32(a: *const f32, b: *const f32, len: u64) -> f32 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }

    let len = len as usize;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = f32x8::ZERO;

        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::new([
                *a.add(offset), *a.add(offset + 1), *a.add(offset + 2), *a.add(offset + 3),
                *a.add(offset + 4), *a.add(offset + 5), *a.add(offset + 6), *a.add(offset + 7),
            ]);
            let vb = f32x8::new([
                *b.add(offset), *b.add(offset + 1), *b.add(offset + 2), *b.add(offset + 3),
                *b.add(offset + 4), *b.add(offset + 5), *b.add(offset + 6), *b.add(offset + 7),
            ]);
            let diff = va - vb;
            sum += diff * diff;
        }

        let arr = sum.to_array();
        let mut result = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

        let base = chunks * 8;
        for i in 0..remainder {
            let diff = *a.add(base + i) - *b.add(base + i);
            result += diff * diff;
        }

        result.sqrt()
    }
}

/// Squared Euclidean distance: sum((a[i] - b[i])^2) - avoids sqrt
#[no_mangle]
pub extern "C" fn vec_euclidean_sq_f32(a: *const f32, b: *const f32, len: u64) -> f32 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }

    let len = len as usize;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = f32x8::ZERO;

        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::new([
                *a.add(offset), *a.add(offset + 1), *a.add(offset + 2), *a.add(offset + 3),
                *a.add(offset + 4), *a.add(offset + 5), *a.add(offset + 6), *a.add(offset + 7),
            ]);
            let vb = f32x8::new([
                *b.add(offset), *b.add(offset + 1), *b.add(offset + 2), *b.add(offset + 3),
                *b.add(offset + 4), *b.add(offset + 5), *b.add(offset + 6), *b.add(offset + 7),
            ]);
            let diff = va - vb;
            sum += diff * diff;
        }

        let arr = sum.to_array();
        let mut result = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

        let base = chunks * 8;
        for i in 0..remainder {
            let diff = *a.add(base + i) - *b.add(base + i);
            result += diff * diff;
        }

        result
    }
}

/// Manhattan (L1) distance: sum(|a[i] - b[i]|)
#[no_mangle]
pub extern "C" fn vec_manhattan_f32(a: *const f32, b: *const f32, len: u64) -> f32 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }

    let len = len as usize;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = f32x8::ZERO;

        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::new([
                *a.add(offset), *a.add(offset + 1), *a.add(offset + 2), *a.add(offset + 3),
                *a.add(offset + 4), *a.add(offset + 5), *a.add(offset + 6), *a.add(offset + 7),
            ]);
            let vb = f32x8::new([
                *b.add(offset), *b.add(offset + 1), *b.add(offset + 2), *b.add(offset + 3),
                *b.add(offset + 4), *b.add(offset + 5), *b.add(offset + 6), *b.add(offset + 7),
            ]);
            let diff = va - vb;
            sum += diff.abs();
        }

        let arr = sum.to_array();
        let mut result = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

        let base = chunks * 8;
        for i in 0..remainder {
            result += (*a.add(base + i) - *b.add(base + i)).abs();
        }

        result
    }
}

/// Cosine similarity: dot(a,b) / (norm(a) * norm(b))
#[no_mangle]
pub extern "C" fn vec_cosine_similarity_f32(a: *const f32, b: *const f32, len: u64) -> f32 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }

    let dot = vec_dot_product_f32(a, b, len);
    let norm_a_sq = vec_dot_product_f32(a, a, len);
    let norm_b_sq = vec_dot_product_f32(b, b, len);

    let denom = (norm_a_sq * norm_b_sq).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

/// L2 normalize in place: data[i] /= norm(data)
#[no_mangle]
pub extern "C" fn vec_l2_normalize_f32(data: *mut f32, len: u64) {
    if data.is_null() || len == 0 {
        return;
    }

    let norm_sq = vec_dot_product_f32(data, data, len);
    let norm = norm_sq.sqrt();

    if norm > 1e-10 {
        vec_scale_f32(data, 1.0 / norm, len);
    }
}

/// Find max value and its index
#[no_mangle]
pub extern "C" fn vec_argmax_with_val_f32(data: *const f32, len: u64, out_idx: *mut u64, out_val: *mut f32) {
    if data.is_null() || len == 0 {
        if !out_idx.is_null() {
            unsafe { *out_idx = 0; }
        }
        if !out_val.is_null() {
            unsafe { *out_val = f32::NEG_INFINITY; }
        }
        return;
    }

    let len = len as usize;
    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;

    unsafe {
        // Process chunks to find max within each chunk, then compare
        let chunks = len / 8;
        let remainder = len % 8;

        for i in 0..chunks {
            let offset = i * 8;
            for j in 0..8 {
                let val = *data.add(offset + j);
                if val > max_val {
                    max_val = val;
                    max_idx = offset + j;
                }
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            let val = *data.add(base + i);
            if val > max_val {
                max_val = val;
                max_idx = base + i;
            }
        }

        if !out_idx.is_null() {
            *out_idx = max_idx as u64;
        }
        if !out_val.is_null() {
            *out_val = max_val;
        }
    }
}
