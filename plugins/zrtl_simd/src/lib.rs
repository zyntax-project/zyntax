//! ZRTL SIMD Plugin
//!
//! High-performance SIMD-accelerated compute kernels with automatic
//! fallback to scalar operations when SIMD is unavailable.
//!
//! ## Features
//! - Runtime CPU feature detection (AVX2, SSE4.1, NEON)
//! - Automatic fallback to scalar code
//! - Safe abstractions over unsafe SIMD intrinsics
//!
//! ## Vector Operations
//! - `$SIMD$dot_product` - Dot product of two vectors
//! - `$SIMD$sum` - Sum all elements
//! - `$SIMD$scale` - Multiply all elements by scalar
//! - `$SIMD$add` - Element-wise addition
//! - `$SIMD$mul` - Element-wise multiplication
//! - `$SIMD$min` - Find minimum value
//! - `$SIMD$max` - Find maximum value
//! - `$SIMD$abs` - Absolute value of all elements
//! - `$SIMD$sqrt` - Square root of all elements
//!
//! ## Matrix Operations
//! - `$SIMD$mat4_mul` - 4x4 matrix multiplication
//! - `$SIMD$mat4_transform` - Transform vec4 by mat4
//!
//! ## Image/Pixel Operations
//! - `$SIMD$blend_rgba` - Alpha blend RGBA pixels
//! - `$SIMD$convert_rgba_bgra` - Convert RGBA to BGRA (and vice versa)
//! - `$SIMD$premultiply_alpha` - Premultiply alpha channel
//!
//! ## Reduction Operations
//! - `$SIMD$horizontal_sum` - Sum with SIMD horizontal add
//! - `$SIMD$count_nonzero` - Count non-zero elements

use zrtl::zrtl_plugin;
use wide::*;

pub use zrtl_simd_kernels::*;

// ============================================================================
// CPU Feature Detection
// ============================================================================

/// Check if AVX2 is available at runtime
#[inline]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Check if SSE4.1 is available at runtime
#[inline]
fn has_sse41() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("sse4.1")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Get SIMD capability level: 0=scalar, 1=SSE4.1, 2=AVX2
#[no_mangle]
pub extern "C" fn simd_capability() -> u32 {
    if has_avx2() {
        2
    } else if has_sse41() {
        1
    } else {
        0
    }
}

// ============================================================================
// Vector Operations (f32)
// ============================================================================




/// Element-wise addition: out[i] = a[i] + b[i]
#[no_mangle]
pub extern "C" fn vec_add_f32(a: *const f32, b: *const f32, out: *mut f32, len: u64) {
    if a.is_null() || b.is_null() || out.is_null() || len == 0 {
        return;
    }

    let len = len as usize;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

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
            let result = va + vb;
            let arr = result.to_array();
            for j in 0..8 {
                *out.add(offset + j) = arr[j];
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            *out.add(base + i) = *a.add(base + i) + *b.add(base + i);
        }
    }
}

/// Element-wise multiplication: out[i] = a[i] * b[i]
#[no_mangle]
pub extern "C" fn vec_mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: u64) {
    if a.is_null() || b.is_null() || out.is_null() || len == 0 {
        return;
    }

    let len = len as usize;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

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
            let result = va * vb;
            let arr = result.to_array();
            for j in 0..8 {
                *out.add(offset + j) = arr[j];
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            *out.add(base + i) = *a.add(base + i) * *b.add(base + i);
        }
    }
}



/// Absolute value: out[i] = |data[i]|
#[no_mangle]
pub extern "C" fn vec_abs_f32(data: *mut f32, len: u64) {
    if data.is_null() || len == 0 {
        return;
    }

    let len = len as usize;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let v = f32x8::new([
                *data.add(offset), *data.add(offset + 1), *data.add(offset + 2), *data.add(offset + 3),
                *data.add(offset + 4), *data.add(offset + 5), *data.add(offset + 6), *data.add(offset + 7),
            ]);
            let result = v.abs();
            let arr = result.to_array();
            for j in 0..8 {
                *data.add(offset + j) = arr[j];
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            *data.add(base + i) = (*data.add(base + i)).abs();
        }
    }
}

/// Square root: out[i] = sqrt(data[i])
#[no_mangle]
pub extern "C" fn vec_sqrt_f32(data: *mut f32, len: u64) {
    if data.is_null() || len == 0 {
        return;
    }

    let len = len as usize;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let v = f32x8::new([
                *data.add(offset), *data.add(offset + 1), *data.add(offset + 2), *data.add(offset + 3),
                *data.add(offset + 4), *data.add(offset + 5), *data.add(offset + 6), *data.add(offset + 7),
            ]);
            let result = v.sqrt();
            let arr = result.to_array();
            for j in 0..8 {
                *data.add(offset + j) = arr[j];
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            *data.add(base + i) = (*data.add(base + i)).sqrt();
        }
    }
}

// ============================================================================
// Matrix Operations (4x4, column-major)
// ============================================================================

/// 4x4 matrix multiplication: out = a * b
/// Matrices are in column-major order (OpenGL convention)
#[no_mangle]
pub extern "C" fn mat4_mul(a: *const f32, b: *const f32, out: *mut f32) {
    if a.is_null() || b.is_null() || out.is_null() {
        return;
    }

    unsafe {
        // Load columns of A
        let a0 = f32x4::new([*a.add(0), *a.add(1), *a.add(2), *a.add(3)]);
        let a1 = f32x4::new([*a.add(4), *a.add(5), *a.add(6), *a.add(7)]);
        let a2 = f32x4::new([*a.add(8), *a.add(9), *a.add(10), *a.add(11)]);
        let a3 = f32x4::new([*a.add(12), *a.add(13), *a.add(14), *a.add(15)]);

        // Compute each column of result
        for col in 0..4 {
            let b_col_base = col * 4;
            let b0 = f32x4::splat(*b.add(b_col_base));
            let b1 = f32x4::splat(*b.add(b_col_base + 1));
            let b2 = f32x4::splat(*b.add(b_col_base + 2));
            let b3 = f32x4::splat(*b.add(b_col_base + 3));

            let result = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            let arr = result.to_array();

            let out_base = col * 4;
            *out.add(out_base) = arr[0];
            *out.add(out_base + 1) = arr[1];
            *out.add(out_base + 2) = arr[2];
            *out.add(out_base + 3) = arr[3];
        }
    }
}

/// Transform a vec4 by a 4x4 matrix: out = mat * vec
#[no_mangle]
pub extern "C" fn mat4_transform_vec4(mat: *const f32, vec: *const f32, out: *mut f32) {
    if mat.is_null() || vec.is_null() || out.is_null() {
        return;
    }

    unsafe {
        let a0 = f32x4::new([*mat.add(0), *mat.add(1), *mat.add(2), *mat.add(3)]);
        let a1 = f32x4::new([*mat.add(4), *mat.add(5), *mat.add(6), *mat.add(7)]);
        let a2 = f32x4::new([*mat.add(8), *mat.add(9), *mat.add(10), *mat.add(11)]);
        let a3 = f32x4::new([*mat.add(12), *mat.add(13), *mat.add(14), *mat.add(15)]);

        let v0 = f32x4::splat(*vec.add(0));
        let v1 = f32x4::splat(*vec.add(1));
        let v2 = f32x4::splat(*vec.add(2));
        let v3 = f32x4::splat(*vec.add(3));

        let result = a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3;
        let arr = result.to_array();

        *out.add(0) = arr[0];
        *out.add(1) = arr[1];
        *out.add(2) = arr[2];
        *out.add(3) = arr[3];
    }
}

/// Transform multiple vec4s by a 4x4 matrix
#[no_mangle]
pub extern "C" fn mat4_transform_vec4_batch(mat: *const f32, vecs: *mut f32, count: u64) {
    if mat.is_null() || vecs.is_null() || count == 0 {
        return;
    }

    unsafe {
        let a0 = f32x4::new([*mat.add(0), *mat.add(1), *mat.add(2), *mat.add(3)]);
        let a1 = f32x4::new([*mat.add(4), *mat.add(5), *mat.add(6), *mat.add(7)]);
        let a2 = f32x4::new([*mat.add(8), *mat.add(9), *mat.add(10), *mat.add(11)]);
        let a3 = f32x4::new([*mat.add(12), *mat.add(13), *mat.add(14), *mat.add(15)]);

        for i in 0..count as usize {
            let base = i * 4;
            let v0 = f32x4::splat(*vecs.add(base));
            let v1 = f32x4::splat(*vecs.add(base + 1));
            let v2 = f32x4::splat(*vecs.add(base + 2));
            let v3 = f32x4::splat(*vecs.add(base + 3));

            let result = a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3;
            let arr = result.to_array();

            *vecs.add(base) = arr[0];
            *vecs.add(base + 1) = arr[1];
            *vecs.add(base + 2) = arr[2];
            *vecs.add(base + 3) = arr[3];
        }
    }
}

// ============================================================================
// Image/Pixel Operations
// ============================================================================

/// Alpha blend RGBA pixels: dst = src * alpha + dst * (1 - alpha)
/// Both src and dst are RGBA u8 arrays
#[no_mangle]
pub extern "C" fn blend_rgba(dst: *mut u8, src: *const u8, count: u64) {
    if dst.is_null() || src.is_null() || count == 0 {
        return;
    }

    unsafe {
        for i in 0..count as usize {
            let base = i * 4;
            let src_r = *src.add(base) as f32;
            let src_g = *src.add(base + 1) as f32;
            let src_b = *src.add(base + 2) as f32;
            let src_a = *src.add(base + 3) as f32 / 255.0;

            let dst_r = *dst.add(base) as f32;
            let dst_g = *dst.add(base + 1) as f32;
            let dst_b = *dst.add(base + 2) as f32;
            let dst_a = *dst.add(base + 3) as f32 / 255.0;

            let inv_src_a = 1.0 - src_a;
            let out_a = src_a + dst_a * inv_src_a;

            if out_a > 0.0 {
                *dst.add(base) = ((src_r * src_a + dst_r * dst_a * inv_src_a) / out_a) as u8;
                *dst.add(base + 1) = ((src_g * src_a + dst_g * dst_a * inv_src_a) / out_a) as u8;
                *dst.add(base + 2) = ((src_b * src_a + dst_b * dst_a * inv_src_a) / out_a) as u8;
                *dst.add(base + 3) = (out_a * 255.0) as u8;
            }
        }
    }
}

/// Convert RGBA to BGRA (swap R and B channels)
/// This is an in-place operation
#[no_mangle]
pub extern "C" fn convert_rgba_bgra(data: *mut u8, count: u64) {
    if data.is_null() || count == 0 {
        return;
    }

    unsafe {
        // Process 2 pixels at a time with SIMD (8 bytes)
        let chunks = count as usize / 2;
        let remainder = count as usize % 2;

        for i in 0..chunks {
            let base = i * 8;
            // Load 8 bytes (2 RGBA pixels)
            let r0 = *data.add(base);
            let b0 = *data.add(base + 2);
            let r1 = *data.add(base + 4);
            let b1 = *data.add(base + 6);

            // Swap R and B
            *data.add(base) = b0;
            *data.add(base + 2) = r0;
            *data.add(base + 4) = b1;
            *data.add(base + 6) = r1;
        }

        // Handle remainder
        let base = chunks * 8;
        for i in 0..remainder {
            let offset = base + i * 4;
            let r = *data.add(offset);
            let b = *data.add(offset + 2);
            *data.add(offset) = b;
            *data.add(offset + 2) = r;
        }
    }
}

/// Premultiply alpha: rgb *= alpha/255
#[no_mangle]
pub extern "C" fn premultiply_alpha(data: *mut u8, count: u64) {
    if data.is_null() || count == 0 {
        return;
    }

    unsafe {
        for i in 0..count as usize {
            let base = i * 4;
            let alpha = *data.add(base + 3) as u32;

            if alpha == 0 {
                *data.add(base) = 0;
                *data.add(base + 1) = 0;
                *data.add(base + 2) = 0;
            } else if alpha < 255 {
                *data.add(base) = ((*data.add(base) as u32 * alpha) / 255) as u8;
                *data.add(base + 1) = ((*data.add(base + 1) as u32 * alpha) / 255) as u8;
                *data.add(base + 2) = ((*data.add(base + 2) as u32 * alpha) / 255) as u8;
            }
        }
    }
}

/// Unpremultiply alpha: rgb /= alpha/255
#[no_mangle]
pub extern "C" fn unpremultiply_alpha(data: *mut u8, count: u64) {
    if data.is_null() || count == 0 {
        return;
    }

    unsafe {
        for i in 0..count as usize {
            let base = i * 4;
            let alpha = *data.add(base + 3) as u32;

            if alpha > 0 && alpha < 255 {
                *data.add(base) = ((*data.add(base) as u32 * 255) / alpha).min(255) as u8;
                *data.add(base + 1) = ((*data.add(base + 1) as u32 * 255) / alpha).min(255) as u8;
                *data.add(base + 2) = ((*data.add(base + 2) as u32 * 255) / alpha).min(255) as u8;
            }
        }
    }
}

// ============================================================================
// Integer Vector Operations
// ============================================================================

/// Sum all elements of an i32 vector
#[no_mangle]
pub extern "C" fn vec_sum_i32(data: *const i32, len: u64) -> i64 {
    if data.is_null() || len == 0 {
        return 0;
    }

    let len = len as usize;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = i32x8::ZERO;

        for i in 0..chunks {
            let offset = i * 8;
            let v = i32x8::new([
                *data.add(offset), *data.add(offset + 1), *data.add(offset + 2), *data.add(offset + 3),
                *data.add(offset + 4), *data.add(offset + 5), *data.add(offset + 6), *data.add(offset + 7),
            ]);
            sum += v;
        }

        let arr = sum.to_array();
        let mut result: i64 = arr[0] as i64 + arr[1] as i64 + arr[2] as i64 + arr[3] as i64
            + arr[4] as i64 + arr[5] as i64 + arr[6] as i64 + arr[7] as i64;

        let base = chunks * 8;
        for i in 0..remainder {
            result += *data.add(base + i) as i64;
        }

        result
    }
}

/// Count non-zero elements
#[no_mangle]
pub extern "C" fn vec_count_nonzero_i32(data: *const i32, len: u64) -> u64 {
    if data.is_null() || len == 0 {
        return 0;
    }

    let len = len as usize;
    let mut count: u64 = 0;

    unsafe {
        for i in 0..len {
            if *data.add(i) != 0 {
                count += 1;
            }
        }
    }

    count
}

// ============================================================================
// Tensor/ML Operations
// ============================================================================

/// General matrix multiplication (GEMM): C = alpha * A @ B + beta * C
/// A is MxK, B is KxN, C is MxN (row-major order)
#[no_mangle]
pub extern "C" fn gemm_f32(
    a: *const f32, b: *const f32, c: *mut f32,
    m: u64, n: u64, k: u64,
    alpha: f32, beta: f32
) {
    if a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;

    unsafe {
        // Simple blocked GEMM with SIMD
        for i in 0..m {
            for j in 0..n {
                // Vectorized dot product for row i of A with column j of B
                let chunks = k / 8;
                let remainder = k % 8;

                let mut simd_sum = f32x8::ZERO;
                for kk in 0..chunks {
                    let offset = kk * 8;
                    let va = f32x8::new([
                        *a.add(i * k + offset),
                        *a.add(i * k + offset + 1),
                        *a.add(i * k + offset + 2),
                        *a.add(i * k + offset + 3),
                        *a.add(i * k + offset + 4),
                        *a.add(i * k + offset + 5),
                        *a.add(i * k + offset + 6),
                        *a.add(i * k + offset + 7),
                    ]);
                    let vb = f32x8::new([
                        *b.add(offset * n + j),
                        *b.add((offset + 1) * n + j),
                        *b.add((offset + 2) * n + j),
                        *b.add((offset + 3) * n + j),
                        *b.add((offset + 4) * n + j),
                        *b.add((offset + 5) * n + j),
                        *b.add((offset + 6) * n + j),
                        *b.add((offset + 7) * n + j),
                    ]);
                    simd_sum += va * vb;
                }

                let arr = simd_sum.to_array();
                let mut sum = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

                // Handle remainder
                let base = chunks * 8;
                for kk in 0..remainder {
                    sum += *a.add(i * k + base + kk) * *b.add((base + kk) * n + j);
                }

                // C = alpha * A @ B + beta * C
                let c_idx = i * n + j;
                *c.add(c_idx) = alpha * sum + beta * *c.add(c_idx);
            }
        }
    }
}

/// ReLU activation: out[i] = max(0, x[i])
#[no_mangle]
pub extern "C" fn relu_f32(data: *mut f32, len: u64) {
    if data.is_null() || len == 0 {
        return;
    }

    let len = len as usize;
    let zero = f32x8::ZERO;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let v = f32x8::new([
                *data.add(offset), *data.add(offset + 1),
                *data.add(offset + 2), *data.add(offset + 3),
                *data.add(offset + 4), *data.add(offset + 5),
                *data.add(offset + 6), *data.add(offset + 7),
            ]);
            let result = v.max(zero);
            let arr = result.to_array();
            for j in 0..8 {
                *data.add(offset + j) = arr[j];
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            let val = *data.add(base + i);
            *data.add(base + i) = if val > 0.0 { val } else { 0.0 };
        }
    }
}

/// Leaky ReLU: out[i] = x[i] if x[i] > 0 else alpha * x[i]
#[no_mangle]
pub extern "C" fn leaky_relu_f32(data: *mut f32, len: u64, alpha: f32) {
    if data.is_null() || len == 0 {
        return;
    }

    let len = len as usize;

    unsafe {
        for i in 0..len {
            let val = *data.add(i);
            *data.add(i) = if val > 0.0 { val } else { alpha * val };
        }
    }
}

/// Sigmoid activation: out[i] = 1 / (1 + exp(-x[i]))
#[no_mangle]
pub extern "C" fn sigmoid_f32(data: *mut f32, len: u64) {
    if data.is_null() || len == 0 {
        return;
    }

    let len = len as usize;

    unsafe {
        for i in 0..len {
            let val = *data.add(i);
            *data.add(i) = 1.0 / (1.0 + (-val).exp());
        }
    }
}

/// Tanh activation: out[i] = tanh(x[i])
#[no_mangle]
pub extern "C" fn tanh_f32(data: *mut f32, len: u64) {
    if data.is_null() || len == 0 {
        return;
    }

    let len = len as usize;

    unsafe {
        for i in 0..len {
            *data.add(i) = (*data.add(i)).tanh();
        }
    }
}

/// Softmax: out[i] = exp(x[i]) / sum(exp(x))
/// Operates in-place, numerically stable version
#[no_mangle]
pub extern "C" fn softmax_f32(data: *mut f32, len: u64) {
    if data.is_null() || len == 0 {
        return;
    }

    let len = len as usize;

    unsafe {
        // Find max for numerical stability
        let mut max_val = *data;
        for i in 1..len {
            let val = *data.add(i);
            if val > max_val {
                max_val = val;
            }
        }

        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for i in 0..len {
            let exp_val = (*data.add(i) - max_val).exp();
            *data.add(i) = exp_val;
            sum += exp_val;
        }

        // Normalize
        if sum > 0.0 {
            for i in 0..len {
                *data.add(i) /= sum;
            }
        }
    }
}

/// Layer normalization: normalize over last dimension
/// out = (x - mean) / sqrt(var + eps) * gamma + beta
#[no_mangle]
pub extern "C" fn layer_norm_f32(
    data: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    batch_size: u64,
    hidden_size: u64,
    eps: f32
) {
    if data.is_null() {
        return;
    }

    let batch_size = batch_size as usize;
    let hidden_size = hidden_size as usize;

    unsafe {
        for b in 0..batch_size {
            let offset = b * hidden_size;

            // Compute mean
            let mut mean = 0.0f32;
            for i in 0..hidden_size {
                mean += *data.add(offset + i);
            }
            mean /= hidden_size as f32;

            // Compute variance
            let mut var = 0.0f32;
            for i in 0..hidden_size {
                let diff = *data.add(offset + i) - mean;
                var += diff * diff;
            }
            var /= hidden_size as f32;

            // Normalize
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..hidden_size {
                let normalized = (*data.add(offset + i) - mean) * inv_std;
                let scaled = if !gamma.is_null() {
                    normalized * *gamma.add(i)
                } else {
                    normalized
                };
                *data.add(offset + i) = if !beta.is_null() {
                    scaled + *beta.add(i)
                } else {
                    scaled
                };
            }
        }
    }
}

/// Batch normalization (inference mode)
/// out = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
#[no_mangle]
pub extern "C" fn batch_norm_f32(
    data: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    running_mean: *const f32,
    running_var: *const f32,
    batch_size: u64,
    channels: u64,
    spatial_size: u64,
    eps: f32
) {
    if data.is_null() || running_mean.is_null() || running_var.is_null() {
        return;
    }

    let batch_size = batch_size as usize;
    let channels = channels as usize;
    let spatial_size = spatial_size as usize;

    unsafe {
        for b in 0..batch_size {
            for c in 0..channels {
                let mean = *running_mean.add(c);
                let var = *running_var.add(c);
                let inv_std = 1.0 / (var + eps).sqrt();

                let g = if !gamma.is_null() { *gamma.add(c) } else { 1.0 };
                let be = if !beta.is_null() { *beta.add(c) } else { 0.0 };

                for s in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + s;
                    *data.add(idx) = (*data.add(idx) - mean) * inv_std * g + be;
                }
            }
        }
    }
}

/// 2D Convolution (NCHW format, no padding, stride=1)
/// Simple implementation for small kernels
#[no_mangle]
pub extern "C" fn conv2d_f32(
    input: *const f32,
    kernel: *const f32,
    output: *mut f32,
    batch: u64,
    in_channels: u64,
    out_channels: u64,
    height: u64,
    width: u64,
    kernel_h: u64,
    kernel_w: u64
) {
    if input.is_null() || kernel.is_null() || output.is_null() {
        return;
    }

    let batch = batch as usize;
    let in_c = in_channels as usize;
    let out_c = out_channels as usize;
    let h = height as usize;
    let w = width as usize;
    let kh = kernel_h as usize;
    let kw = kernel_w as usize;

    let out_h = h - kh + 1;
    let out_w = w - kw + 1;

    unsafe {
        for b in 0..batch {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;

                        for ic in 0..in_c {
                            for khi in 0..kh {
                                for kwi in 0..kw {
                                    let in_idx = b * in_c * h * w
                                        + ic * h * w
                                        + (oh + khi) * w
                                        + (ow + kwi);
                                    let k_idx = oc * in_c * kh * kw
                                        + ic * kh * kw
                                        + khi * kw
                                        + kwi;
                                    sum += *input.add(in_idx) * *kernel.add(k_idx);
                                }
                            }
                        }

                        let out_idx = b * out_c * out_h * out_w
                            + oc * out_h * out_w
                            + oh * out_w
                            + ow;
                        *output.add(out_idx) = sum;
                    }
                }
            }
        }
    }
}

/// Max pooling 2D (NCHW format)
#[no_mangle]
pub extern "C" fn max_pool2d_f32(
    input: *const f32,
    output: *mut f32,
    batch: u64,
    channels: u64,
    height: u64,
    width: u64,
    pool_h: u64,
    pool_w: u64
) {
    if input.is_null() || output.is_null() {
        return;
    }

    let batch = batch as usize;
    let c = channels as usize;
    let h = height as usize;
    let w = width as usize;
    let ph = pool_h as usize;
    let pw = pool_w as usize;

    let out_h = h / ph;
    let out_w = w / pw;

    unsafe {
        for b in 0..batch {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;

                        for phi in 0..ph {
                            for pwi in 0..pw {
                                let in_idx = b * c * h * w
                                    + ch * h * w
                                    + (oh * ph + phi) * w
                                    + (ow * pw + pwi);
                                let val = *input.add(in_idx);
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }

                        let out_idx = b * c * out_h * out_w
                            + ch * out_h * out_w
                            + oh * out_w
                            + ow;
                        *output.add(out_idx) = max_val;
                    }
                }
            }
        }
    }
}

/// Average pooling 2D (NCHW format)
#[no_mangle]
pub extern "C" fn avg_pool2d_f32(
    input: *const f32,
    output: *mut f32,
    batch: u64,
    channels: u64,
    height: u64,
    width: u64,
    pool_h: u64,
    pool_w: u64
) {
    if input.is_null() || output.is_null() {
        return;
    }

    let batch = batch as usize;
    let c = channels as usize;
    let h = height as usize;
    let w = width as usize;
    let ph = pool_h as usize;
    let pw = pool_w as usize;

    let out_h = h / ph;
    let out_w = w / pw;
    let pool_size = (ph * pw) as f32;

    unsafe {
        for b in 0..batch {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;

                        for phi in 0..ph {
                            for pwi in 0..pw {
                                let in_idx = b * c * h * w
                                    + ch * h * w
                                    + (oh * ph + phi) * w
                                    + (ow * pw + pwi);
                                sum += *input.add(in_idx);
                            }
                        }

                        let out_idx = b * c * out_h * out_w
                            + ch * out_h * out_w
                            + oh * out_w
                            + ow;
                        *output.add(out_idx) = sum / pool_size;
                    }
                }
            }
        }
    }
}

/// Dropout (inference mode - just pass through, training would scale)
/// For inference, this is a no-op since we don't drop during inference
#[no_mangle]
pub extern "C" fn dropout_f32(_data: *mut f32, _len: u64, _p: f32) {
    // No-op for inference
}

/// Cross-entropy loss (after softmax)
/// Returns -sum(target * log(pred))
#[no_mangle]
pub extern "C" fn cross_entropy_loss_f32(
    pred: *const f32,
    target: *const f32,
    len: u64
) -> f32 {
    if pred.is_null() || target.is_null() || len == 0 {
        return 0.0;
    }

    let len = len as usize;
    let mut loss = 0.0f32;

    unsafe {
        for i in 0..len {
            let p = (*pred.add(i)).max(1e-7); // Avoid log(0)
            loss -= *target.add(i) * p.ln();
        }
    }

    loss
}

/// Argmax: returns index of maximum value
#[no_mangle]
pub extern "C" fn argmax_f32(data: *const f32, len: u64) -> u64 {
    if data.is_null() || len == 0 {
        return 0;
    }

    let len = len as usize;
    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;

    unsafe {
        for i in 0..len {
            let val = *data.add(i);
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
    }

    max_idx as u64
}

/// Clip values to range [min, max]
#[no_mangle]
pub extern "C" fn clip_f32(data: *mut f32, len: u64, min_val: f32, max_val: f32) {
    if data.is_null() || len == 0 {
        return;
    }

    let len = len as usize;
    let min_vec = f32x8::splat(min_val);
    let max_vec = f32x8::splat(max_val);

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let v = f32x8::new([
                *data.add(offset), *data.add(offset + 1),
                *data.add(offset + 2), *data.add(offset + 3),
                *data.add(offset + 4), *data.add(offset + 5),
                *data.add(offset + 6), *data.add(offset + 7),
            ]);
            let result = v.max(min_vec).min(max_vec);
            let arr = result.to_array();
            for j in 0..8 {
                *data.add(offset + j) = arr[j];
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            let val = *data.add(base + i);
            *data.add(base + i) = val.max(min_val).min(max_val);
        }
    }
}

/// Transpose 2D matrix: out[j,i] = in[i,j]
#[no_mangle]
pub extern "C" fn transpose_2d_f32(
    input: *const f32,
    output: *mut f32,
    rows: u64,
    cols: u64
) {
    if input.is_null() || output.is_null() {
        return;
    }

    let rows = rows as usize;
    let cols = cols as usize;

    unsafe {
        for i in 0..rows {
            for j in 0..cols {
                *output.add(j * rows + i) = *input.add(i * cols + j);
            }
        }
    }
}

/// Element-wise exponential
#[no_mangle]
pub extern "C" fn exp_f32(data: *mut f32, len: u64) {
    if data.is_null() || len == 0 {
        return;
    }

    let len = len as usize;

    unsafe {
        for i in 0..len {
            *data.add(i) = (*data.add(i)).exp();
        }
    }
}

/// Element-wise natural log
#[no_mangle]
pub extern "C" fn log_f32(data: *mut f32, len: u64) {
    if data.is_null() || len == 0 {
        return;
    }

    let len = len as usize;

    unsafe {
        for i in 0..len {
            *data.add(i) = (*data.add(i)).ln();
        }
    }
}

// ============================================================================
// Additional Vector Operations for ML Plugins
// ============================================================================


/// Element-wise subtraction: out[i] = a[i] - b[i]
#[no_mangle]
pub extern "C" fn vec_sub_f32(a: *const f32, b: *const f32, out: *mut f32, len: u64) {
    if a.is_null() || b.is_null() || out.is_null() || len == 0 {
        return;
    }

    let len = len as usize;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

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
            let result = va - vb;
            let arr = result.to_array();
            for j in 0..8 {
                *out.add(offset + j) = arr[j];
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            *out.add(base + i) = *a.add(base + i) - *b.add(base + i);
        }
    }
}




/// Sum of absolute values: sum(|data[i]|)
#[no_mangle]
pub extern "C" fn vec_abs_sum_f32(data: *const f32, len: u64) -> f32 {
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
                *data.add(offset), *data.add(offset + 1), *data.add(offset + 2), *data.add(offset + 3),
                *data.add(offset + 4), *data.add(offset + 5), *data.add(offset + 6), *data.add(offset + 7),
            ]);
            sum += v.abs();
        }

        let arr = sum.to_array();
        let mut result = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

        let base = chunks * 8;
        for i in 0..remainder {
            result += (*data.add(base + i)).abs();
        }

        result
    }
}

/// Fused multiply-add: out[i] = a[i] * b[i] + c[i]
#[no_mangle]
pub extern "C" fn vec_fma_f32(a: *const f32, b: *const f32, c: *const f32, out: *mut f32, len: u64) {
    if a.is_null() || b.is_null() || c.is_null() || out.is_null() || len == 0 {
        return;
    }

    let len = len as usize;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

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
            let vc = f32x8::new([
                *c.add(offset), *c.add(offset + 1), *c.add(offset + 2), *c.add(offset + 3),
                *c.add(offset + 4), *c.add(offset + 5), *c.add(offset + 6), *c.add(offset + 7),
            ]);
            let result = va * vb + vc;
            let arr = result.to_array();
            for j in 0..8 {
                *out.add(offset + j) = arr[j];
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            *out.add(base + i) = *a.add(base + i) * *b.add(base + i) + *c.add(base + i);
        }
    }
}

/// Scalar fused multiply-add: out[i] = a[i] * scalar + b[i]
#[no_mangle]
pub extern "C" fn vec_fma_scalar_f32(a: *const f32, scalar: f32, b: *const f32, out: *mut f32, len: u64) {
    if a.is_null() || b.is_null() || out.is_null() || len == 0 {
        return;
    }

    let len = len as usize;
    let scalar_vec = f32x8::splat(scalar);

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

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
            let result = va * scalar_vec + vb;
            let arr = result.to_array();
            for j in 0..8 {
                *out.add(offset + j) = arr[j];
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            *out.add(base + i) = *a.add(base + i) * scalar + *b.add(base + i);
        }
    }
}




/// Find min value and its index
#[no_mangle]
pub extern "C" fn vec_argmin_with_val_f32(data: *const f32, len: u64, out_idx: *mut u64, out_val: *mut f32) {
    if data.is_null() || len == 0 {
        if !out_idx.is_null() {
            unsafe { *out_idx = 0; }
        }
        if !out_val.is_null() {
            unsafe { *out_val = f32::INFINITY; }
        }
        return;
    }

    let len = len as usize;
    let mut min_idx = 0usize;
    let mut min_val = f32::INFINITY;

    unsafe {
        let chunks = len / 8;
        let remainder = len % 8;

        for i in 0..chunks {
            let offset = i * 8;
            for j in 0..8 {
                let val = *data.add(offset + j);
                if val < min_val {
                    min_val = val;
                    min_idx = offset + j;
                }
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            let val = *data.add(base + i);
            if val < min_val {
                min_val = val;
                min_idx = base + i;
            }
        }

        if !out_idx.is_null() {
            *out_idx = min_idx as u64;
        }
        if !out_val.is_null() {
            *out_val = min_val;
        }
    }
}

// ============================================================================
// Plugin Export
// ============================================================================

zrtl_plugin! {
    name: "zrtl_simd",
    symbols: [
        // Capability detection
        ("$SIMD$capability", simd_capability),

        // Vector f32 operations
        ("$SIMD$dot_product_f32", vec_dot_product_f32),
        ("$SIMD$sum_f32", vec_sum_f32),
        ("$SIMD$scale_f32", vec_scale_f32),
        ("$SIMD$add_f32", vec_add_f32),
        ("$SIMD$sub_f32", vec_sub_f32),
        ("$SIMD$mul_f32", vec_mul_f32),
        ("$SIMD$min_f32", vec_min_f32),
        ("$SIMD$max_f32", vec_max_f32),
        ("$SIMD$abs_f32", vec_abs_f32),
        ("$SIMD$sqrt_f32", vec_sqrt_f32),
        ("$SIMD$fill_f32", vec_fill_f32),
        ("$SIMD$abs_sum_f32", vec_abs_sum_f32),

        // Distance and similarity
        ("$SIMD$euclidean_f32", vec_euclidean_f32),
        ("$SIMD$euclidean_sq_f32", vec_euclidean_sq_f32),
        ("$SIMD$manhattan_f32", vec_manhattan_f32),
        ("$SIMD$cosine_similarity_f32", vec_cosine_similarity_f32),
        ("$SIMD$l2_normalize_f32", vec_l2_normalize_f32),

        // Fused operations
        ("$SIMD$fma_f32", vec_fma_f32),
        ("$SIMD$fma_scalar_f32", vec_fma_scalar_f32),

        // Argmax/Argmin
        ("$SIMD$argmax_with_val_f32", vec_argmax_with_val_f32),
        ("$SIMD$argmin_with_val_f32", vec_argmin_with_val_f32),

        // Matrix operations
        ("$SIMD$mat4_mul", mat4_mul),
        ("$SIMD$mat4_transform_vec4", mat4_transform_vec4),
        ("$SIMD$mat4_transform_batch", mat4_transform_vec4_batch),

        // Pixel operations
        ("$SIMD$blend_rgba", blend_rgba),
        ("$SIMD$convert_rgba_bgra", convert_rgba_bgra),
        ("$SIMD$premultiply_alpha", premultiply_alpha),
        ("$SIMD$unpremultiply_alpha", unpremultiply_alpha),

        // Integer operations
        ("$SIMD$sum_i32", vec_sum_i32),
        ("$SIMD$count_nonzero_i32", vec_count_nonzero_i32),

        // Tensor/ML operations
        ("$SIMD$gemm_f32", gemm_f32),
        ("$SIMD$transpose_2d_f32", transpose_2d_f32),

        // Activation functions
        ("$SIMD$relu_f32", relu_f32),
        ("$SIMD$leaky_relu_f32", leaky_relu_f32),
        ("$SIMD$sigmoid_f32", sigmoid_f32),
        ("$SIMD$tanh_f32", tanh_f32),
        ("$SIMD$softmax_f32", softmax_f32),

        // Normalization
        ("$SIMD$layer_norm_f32", layer_norm_f32),
        ("$SIMD$batch_norm_f32", batch_norm_f32),

        // Convolution and pooling
        ("$SIMD$conv2d_f32", conv2d_f32),
        ("$SIMD$max_pool2d_f32", max_pool2d_f32),
        ("$SIMD$avg_pool2d_f32", avg_pool2d_f32),

        // Utilities
        ("$SIMD$dropout_f32", dropout_f32),
        ("$SIMD$cross_entropy_f32", cross_entropy_loss_f32),
        ("$SIMD$argmax_f32", argmax_f32),
        ("$SIMD$clip_f32", clip_f32),
        ("$SIMD$exp_f32", exp_f32),
        ("$SIMD$log_f32", log_f32),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capability() {
        let cap = simd_capability();
        println!("SIMD capability level: {}", cap);
        assert!(cap <= 2);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = vec_dot_product_f32(a.as_ptr(), b.as_ptr(), a.len() as u64);
        assert_eq!(result, 55.0); // 1+2+3+4+5+6+7+8+9+10
    }

    #[test]
    fn test_sum() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = vec_sum_f32(data.as_ptr(), data.len() as u64);
        assert_eq!(result, 45.0);
    }

    #[test]
    fn test_scale() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        vec_scale_f32(data.as_mut_ptr(), 2.0, data.len() as u64);
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_min_max() {
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];

        let min = vec_min_f32(data.as_ptr(), data.len() as u64);
        let max = vec_max_f32(data.as_ptr(), data.len() as u64);

        assert_eq!(min, 1.0);
        assert_eq!(max, 9.0);
    }

    #[test]
    fn test_mat4_mul_identity() {
        // Identity matrix
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let mut result = [0.0f32; 16];

        mat4_mul(identity.as_ptr(), identity.as_ptr(), result.as_mut_ptr());

        for i in 0..16 {
            assert!((result[i] - identity[i]).abs() < 0.0001);
        }
    }

    #[test]
    fn test_rgba_bgra_conversion() {
        let mut data = vec![255u8, 0, 0, 255, 0, 255, 0, 255]; // Red, Green
        convert_rgba_bgra(data.as_mut_ptr(), 2);
        assert_eq!(data, vec![0, 0, 255, 255, 0, 255, 0, 255]); // Blue, Green
    }

    #[test]
    fn test_premultiply_alpha() {
        let mut data = vec![200u8, 100, 50, 128]; // RGBA with 50% alpha
        premultiply_alpha(data.as_mut_ptr(), 1);

        // 200 * 128 / 255 ≈ 100
        // 100 * 128 / 255 ≈ 50
        // 50 * 128 / 255 ≈ 25
        assert_eq!(data[0], 100);
        assert_eq!(data[1], 50);
        assert_eq!(data[2], 25);
        assert_eq!(data[3], 128); // Alpha unchanged
    }

    // ML operation tests

    #[test]
    fn test_relu() {
        let mut data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5];
        relu_f32(data.as_mut_ptr(), data.len() as u64);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.5]);
    }

    #[test]
    fn test_leaky_relu() {
        let mut data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        leaky_relu_f32(data.as_mut_ptr(), data.len() as u64, 0.1);
        assert!((data[0] - (-0.2)).abs() < 0.0001);
        assert!((data[1] - (-0.1)).abs() < 0.0001);
        assert_eq!(data[2], 0.0);
        assert_eq!(data[3], 1.0);
        assert_eq!(data[4], 2.0);
    }

    #[test]
    fn test_sigmoid() {
        let mut data = vec![0.0f32, 1.0, -1.0, 10.0, -10.0];
        sigmoid_f32(data.as_mut_ptr(), data.len() as u64);

        // sigmoid(0) = 0.5
        assert!((data[0] - 0.5).abs() < 0.0001);
        // sigmoid(large positive) ≈ 1
        assert!(data[3] > 0.99);
        // sigmoid(large negative) ≈ 0
        assert!(data[4] < 0.01);
    }

    #[test]
    fn test_softmax() {
        let mut data = vec![1.0f32, 2.0, 3.0];
        softmax_f32(data.as_mut_ptr(), data.len() as u64);

        // Sum should be 1.0
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.0001);

        // Largest input should have largest probability
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_argmax() {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
        let result = argmax_f32(data.as_ptr(), data.len() as u64);
        assert_eq!(result, 1); // Index of 5.0
    }

    #[test]
    fn test_clip() {
        let mut data = vec![-2.0f32, -0.5, 0.0, 0.5, 2.0];
        clip_f32(data.as_mut_ptr(), data.len() as u64, -1.0, 1.0);
        assert_eq!(data, vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_gemm_identity() {
        // Test: A @ I = A
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let identity: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let mut c: Vec<f32> = vec![0.0; 4];

        gemm_f32(
            a.as_ptr(), identity.as_ptr(), c.as_mut_ptr(),
            2, 2, 2, // m, n, k
            1.0, 0.0 // alpha, beta
        );

        assert!((c[0] - 1.0).abs() < 0.0001);
        assert!((c[1] - 2.0).abs() < 0.0001);
        assert!((c[2] - 3.0).abs() < 0.0001);
        assert!((c[3] - 4.0).abs() < 0.0001);
    }

    #[test]
    fn test_transpose_2d() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let mut output = vec![0.0f32; 6];
        transpose_2d_f32(data.as_ptr(), output.as_mut_ptr(), 2, 3);
        // Should be 3x2: [[1,4], [2,5], [3,6]]
        assert_eq!(output, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // Tests for new SIMD functions

    #[test]
    fn test_fill() {
        let mut data = vec![0.0f32; 10];
        vec_fill_f32(data.as_mut_ptr(), 3.14, 10);
        for v in &data {
            assert!((v - 3.14).abs() < 0.0001);
        }
    }

    #[test]
    fn test_sub() {
        let a = vec![5.0f32, 4.0, 3.0, 2.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        vec_sub_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4);
        assert_eq!(out, vec![4.0, 2.0, 0.0, -2.0]);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![3.0f32, 4.0, 0.0];
        let dist = vec_euclidean_f32(a.as_ptr(), b.as_ptr(), 3);
        assert!((dist - 5.0).abs() < 0.0001); // 3-4-5 triangle
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![3.0f32, 4.0, 5.0];
        let dist = vec_manhattan_f32(a.as_ptr(), b.as_ptr(), 3);
        assert!((dist - 12.0).abs() < 0.0001); // 3 + 4 + 5
    }

    #[test]
    fn test_abs_sum() {
        let data = vec![-1.0f32, 2.0, -3.0, 4.0];
        let sum = vec_abs_sum_f32(data.as_ptr(), 4);
        assert!((sum - 10.0).abs() < 0.0001); // 1 + 2 + 3 + 4
    }

    #[test]
    fn test_fma() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![2.0f32, 2.0, 2.0, 2.0];
        let c = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut out = vec![0.0f32; 4];
        vec_fma_f32(a.as_ptr(), b.as_ptr(), c.as_ptr(), out.as_mut_ptr(), 4);
        assert_eq!(out, vec![3.0, 5.0, 7.0, 9.0]); // a*b + c
    }

    #[test]
    fn test_cosine_similarity() {
        // Same direction vectors
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![2.0f32, 0.0, 0.0];
        let sim = vec_cosine_similarity_f32(a.as_ptr(), b.as_ptr(), 3);
        assert!((sim - 1.0).abs() < 0.0001);

        // Orthogonal vectors
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let sim = vec_cosine_similarity_f32(a.as_ptr(), b.as_ptr(), 3);
        assert!(sim.abs() < 0.0001);
    }

    #[test]
    fn test_l2_normalize() {
        let mut data = vec![3.0f32, 4.0];
        vec_l2_normalize_f32(data.as_mut_ptr(), 2);
        // norm = 5, so [0.6, 0.8]
        assert!((data[0] - 0.6).abs() < 0.0001);
        assert!((data[1] - 0.8).abs() < 0.0001);
    }

    #[test]
    fn test_argmax_with_val() {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
        let mut idx: u64 = 0;
        let mut val: f32 = 0.0;
        vec_argmax_with_val_f32(data.as_ptr(), 5, &mut idx, &mut val);
        assert_eq!(idx, 1);
        assert!((val - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_argmin_with_val() {
        let data = vec![3.0f32, 5.0, 1.0, 2.0, 4.0];
        let mut idx: u64 = 0;
        let mut val: f32 = 0.0;
        vec_argmin_with_val_f32(data.as_ptr(), 5, &mut idx, &mut val);
        assert_eq!(idx, 2);
        assert!((val - 1.0).abs() < 0.0001);
    }
}
