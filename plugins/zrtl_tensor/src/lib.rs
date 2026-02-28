//! # ZRTL Tensor Plugin
//!
//! Provides tensor data structures and operations for machine learning workloads.
//! This is the foundational data structure for ZynML.
//!
//! ## Features
//!
//! - Multi-dimensional tensor with arbitrary shapes
//! - Multiple data types (f32, f64, i32, i64, etc.)
//! - Memory-efficient views and slicing
//! - Broadcasting support for element-wise operations
//! - Reference counting for shared tensors
//!
//! ## Memory Layout
//!
//! Tensors use a contiguous memory layout with row-major (C) ordering by default.
//! Strides allow for efficient views without data copying.

use std::alloc::{alloc_zeroed, dealloc, Layout as AllocLayout};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use zrtl::zrtl_plugin;

// Import SIMD functions for optimized operations
use zrtl_simd::{
    vec_fill_f32, vec_sum_f32, vec_max_f32, vec_min_f32,
    vec_argmax_with_val_f32,
};

// ============================================================================
// Data Types
// ============================================================================

/// Tensor data type
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32 = 0,
    F64 = 1,
    I8 = 2,
    I16 = 3,
    I32 = 4,
    I64 = 5,
    U8 = 6,
    U16 = 7,
    U32 = 8,
    U64 = 9,
    Bool = 10,
    F16 = 11,  // Half precision (stored as u16)
    BF16 = 12, // Brain float16 (stored as u16)
}

impl DType {
    /// Get the size of this data type in bytes
    pub fn size_bytes(self) -> usize {
        match self {
            DType::Bool | DType::I8 | DType::U8 => 1,
            DType::I16 | DType::U16 | DType::F16 | DType::BF16 => 2,
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::I64 | DType::U64 | DType::F64 => 8,
        }
    }

    /// Get the alignment of this data type
    pub fn alignment(self) -> usize {
        self.size_bytes()
    }
}

/// Memory layout ordering
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    RowMajor = 0,    // C-style, last dimension varies fastest
    ColumnMajor = 1, // Fortran-style, first dimension varies fastest
}

// ============================================================================
// Tensor Handle
// ============================================================================

/// Maximum number of dimensions supported
pub const MAX_DIMS: usize = 8;

/// Internal tensor storage with reference counting
struct TensorStorage {
    /// Reference count
    refcount: AtomicUsize,
    /// Data pointer (owned)
    data: *mut u8,
    /// Total size in bytes
    size_bytes: usize,
    /// Memory layout for deallocation
    alloc_layout: AllocLayout,
}

impl TensorStorage {
    fn new(size_bytes: usize, dtype: DType) -> Option<*mut Self> {
        if size_bytes == 0 {
            return None;
        }

        let align = dtype.alignment().max(8);
        let alloc_layout = AllocLayout::from_size_align(size_bytes, align).ok()?;

        let data = unsafe { alloc_zeroed(alloc_layout) };
        if data.is_null() {
            return None;
        }

        let storage = Box::new(TensorStorage {
            refcount: AtomicUsize::new(1),
            data,
            size_bytes,
            alloc_layout,
        });

        Some(Box::into_raw(storage))
    }

    fn increment_refcount(ptr: *mut Self) {
        if !ptr.is_null() {
            unsafe {
                (*ptr).refcount.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn decrement_refcount(ptr: *mut Self) -> bool {
        if ptr.is_null() {
            return false;
        }

        unsafe {
            if (*ptr).refcount.fetch_sub(1, Ordering::Release) == 1 {
                std::sync::atomic::fence(Ordering::Acquire);
                // Free the data
                dealloc((*ptr).data, (*ptr).alloc_layout);
                // Free the storage struct
                drop(Box::from_raw(ptr));
                return true;
            }
        }
        false
    }
}

/// Tensor handle - C ABI compatible
#[repr(C)]
pub struct TensorHandle {
    /// Pointer to storage (reference counted)
    storage: *mut TensorStorage,
    /// Data pointer (may be offset for views)
    data: *mut u8,
    /// Shape array
    shape: [usize; MAX_DIMS],
    /// Strides array (in elements, not bytes)
    strides: [isize; MAX_DIMS],
    /// Number of dimensions
    ndim: u32,
    /// Data type
    dtype: DType,
    /// Memory layout
    layout: MemoryLayout,
    /// Flags
    flags: u32,
}

// Tensor flags
const FLAG_CONTIGUOUS: u32 = 1 << 0;
const FLAG_OWNS_DATA: u32 = 1 << 1;
const FLAG_WRITEABLE: u32 = 1 << 2;

impl TensorHandle {
    /// Check if tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        self.flags & FLAG_CONTIGUOUS != 0
    }

    /// Check if tensor owns its data
    pub fn owns_data(&self) -> bool {
        self.flags & FLAG_OWNS_DATA != 0
    }

    /// Check if tensor is writeable
    pub fn is_writeable(&self) -> bool {
        self.flags & FLAG_WRITEABLE != 0
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        let mut n = 1usize;
        for i in 0..self.ndim as usize {
            n = n.saturating_mul(self.shape[i]);
        }
        n
    }

    /// Get the total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }

    /// Calculate strides for a contiguous tensor
    fn calculate_strides(shape: &[usize], layout: MemoryLayout) -> [isize; MAX_DIMS] {
        let mut strides = [0isize; MAX_DIMS];
        let ndim = shape.len();

        if ndim == 0 {
            return strides;
        }

        match layout {
            MemoryLayout::RowMajor => {
                strides[ndim - 1] = 1;
                for i in (0..ndim - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1] as isize;
                }
            }
            MemoryLayout::ColumnMajor => {
                strides[0] = 1;
                for i in 1..ndim {
                    strides[i] = strides[i - 1] * shape[i - 1] as isize;
                }
            }
        }

        strides
    }

    /// Check if strides represent contiguous memory
    fn check_contiguous(shape: &[usize], strides: &[isize], layout: MemoryLayout) -> bool {
        let expected = Self::calculate_strides(shape, layout);
        for i in 0..shape.len() {
            if strides[i] != expected[i] {
                return false;
            }
        }
        true
    }
}

// ============================================================================
// Handle Type for FFI
// ============================================================================

/// Opaque handle type for FFI
pub type TensorPtr = *mut TensorHandle;

/// Null tensor constant
pub const TENSOR_NULL: TensorPtr = std::ptr::null_mut();

// ============================================================================
// Creation Functions
// ============================================================================

/// Create a new tensor with uninitialized data
#[no_mangle]
pub extern "C" fn tensor_new(
    shape_ptr: *const usize,
    ndim: u32,
    dtype: u8,
) -> TensorPtr {
    if shape_ptr.is_null() || ndim == 0 || ndim as usize > MAX_DIMS {
        return TENSOR_NULL;
    }

    let dtype = match dtype {
        0 => DType::F32,
        1 => DType::F64,
        2 => DType::I8,
        3 => DType::I16,
        4 => DType::I32,
        5 => DType::I64,
        6 => DType::U8,
        7 => DType::U16,
        8 => DType::U32,
        9 => DType::U64,
        10 => DType::Bool,
        11 => DType::F16,
        12 => DType::BF16,
        _ => return TENSOR_NULL,
    };

    // Copy shape
    let mut shape = [0usize; MAX_DIMS];
    let mut numel = 1usize;
    for i in 0..ndim as usize {
        unsafe {
            shape[i] = *shape_ptr.add(i);
            numel = numel.saturating_mul(shape[i]);
        }
    }

    if numel == 0 {
        return TENSOR_NULL;
    }

    let size_bytes = numel * dtype.size_bytes();
    let storage = match TensorStorage::new(size_bytes, dtype) {
        Some(s) => s,
        None => return TENSOR_NULL,
    };

    let strides = TensorHandle::calculate_strides(&shape[..ndim as usize], MemoryLayout::RowMajor);

    let handle = Box::new(TensorHandle {
        storage,
        data: unsafe { (*storage).data },
        shape,
        strides,
        ndim,
        dtype,
        layout: MemoryLayout::RowMajor,
        flags: FLAG_CONTIGUOUS | FLAG_OWNS_DATA | FLAG_WRITEABLE,
    });

    Box::into_raw(handle)
}

/// Create a tensor filled with zeros
#[no_mangle]
pub extern "C" fn tensor_zeros(
    shape_ptr: *const usize,
    ndim: u32,
    dtype: u8,
) -> TensorPtr {
    // tensor_new already uses alloc_zeroed
    tensor_new(shape_ptr, ndim, dtype)
}

/// Create a tensor filled with ones
#[no_mangle]
pub extern "C" fn tensor_ones(
    shape_ptr: *const usize,
    ndim: u32,
    dtype: u8,
) -> TensorPtr {
    let tensor = tensor_new(shape_ptr, ndim, dtype);
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    unsafe {
        let t = &*tensor;
        let numel = t.numel();

        match t.dtype {
            DType::F32 => {
                // Use SIMD-optimized fill
                vec_fill_f32(t.data as *mut f32, 1.0, numel as u64);
            }
            DType::F64 => {
                let ptr = t.data as *mut f64;
                for i in 0..numel {
                    *ptr.add(i) = 1.0;
                }
            }
            DType::I32 => {
                let ptr = t.data as *mut i32;
                for i in 0..numel {
                    *ptr.add(i) = 1;
                }
            }
            DType::I64 => {
                let ptr = t.data as *mut i64;
                for i in 0..numel {
                    *ptr.add(i) = 1;
                }
            }
            DType::U8 | DType::Bool => {
                // Use memset for single-byte types
                ptr::write_bytes(t.data, 1, numel);
            }
            _ => {
                // For other types, write 1 as bytes
                let elem_size = t.dtype.size_bytes();
                for i in 0..numel {
                    let ptr = t.data.add(i * elem_size);
                    match elem_size {
                        1 => *ptr = 1,
                        2 => *(ptr as *mut u16) = 1,
                        4 => *(ptr as *mut u32) = 1,
                        8 => *(ptr as *mut u64) = 1,
                        _ => {}
                    }
                }
            }
        }
    }

    tensor
}

/// Create a tensor filled with zeros (f32 dtype, no dtype parameter needed)
#[no_mangle]
pub extern "C" fn tensor_zeros_f32(
    shape_ptr: *const usize,
    ndim: u32,
) -> TensorPtr {
    tensor_zeros(shape_ptr, ndim, DType::F32 as u8)
}

/// Create a tensor filled with ones (f32 dtype, no dtype parameter needed)
#[no_mangle]
pub extern "C" fn tensor_ones_f32(
    shape_ptr: *const usize,
    ndim: u32,
) -> TensorPtr {
    tensor_ones(shape_ptr, ndim, DType::F32 as u8)
}

/// Create a 1D tensor filled with zeros (convenience, no pointer/List needed)
#[no_mangle]
pub extern "C" fn tensor_zeros_1d(n: i64) -> TensorPtr {
    let shape = [n as usize];
    tensor_zeros(shape.as_ptr(), 1, DType::F32 as u8)
}

/// Create a 2D tensor filled with zeros (convenience, no pointer/List needed)
#[no_mangle]
pub extern "C" fn tensor_zeros_2d(rows: i64, cols: i64) -> TensorPtr {
    let shape = [rows as usize, cols as usize];
    tensor_zeros(shape.as_ptr(), 2, DType::F32 as u8)
}

/// Create a 3D tensor filled with zeros (convenience, no pointer/List needed)
#[no_mangle]
pub extern "C" fn tensor_zeros_3d(d0: i64, d1: i64, d2: i64) -> TensorPtr {
    let shape = [d0 as usize, d1 as usize, d2 as usize];
    tensor_zeros(shape.as_ptr(), 3, DType::F32 as u8)
}

/// Create a 1D tensor filled with ones (convenience, no pointer/List needed)
#[no_mangle]
pub extern "C" fn tensor_ones_1d(n: i64) -> TensorPtr {
    let shape = [n as usize];
    tensor_ones(shape.as_ptr(), 1, DType::F32 as u8)
}

/// Create a 2D tensor filled with ones (convenience, no pointer/List needed)
#[no_mangle]
pub extern "C" fn tensor_ones_2d(rows: i64, cols: i64) -> TensorPtr {
    let shape = [rows as usize, cols as usize];
    tensor_ones(shape.as_ptr(), 2, DType::F32 as u8)
}

/// Create a 3D tensor filled with ones (convenience, no pointer/List needed)
#[no_mangle]
pub extern "C" fn tensor_ones_3d(d0: i64, d1: i64, d2: i64) -> TensorPtr {
    let shape = [d0 as usize, d1 as usize, d2 as usize];
    tensor_ones(shape.as_ptr(), 3, DType::F32 as u8)
}

/// Create a tensor filled with a scalar value
#[no_mangle]
pub extern "C" fn tensor_full_f32(
    shape_ptr: *const usize,
    ndim: u32,
    value: f32,
) -> TensorPtr {
    let tensor = tensor_new(shape_ptr, ndim, DType::F32 as u8);
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    unsafe {
        let t = &*tensor;
        let numel = t.numel();
        // Use SIMD-optimized fill
        vec_fill_f32(t.data as *mut f32, value, numel as u64);
    }

    tensor
}

/// Create a tensor from a raw f32 pointer and count
///
/// # Arguments
/// * `data` - Pointer to f32 array data
/// * `count` - Number of elements
///
/// # Returns
/// A 1D tensor containing a copy of the data
///
/// # Safety
/// The caller must ensure `data` points to at least `count` f32 values
#[no_mangle]
pub extern "C" fn tensor_from_raw_f32(data: *const f32, count: usize) -> TensorPtr {
    if data.is_null() || count == 0 {
        return TENSOR_NULL;
    }

    unsafe {
        let shape = [count];
        let tensor = tensor_new(shape.as_ptr(), 1, DType::F32 as u8);
        if tensor.is_null() {
            return TENSOR_NULL;
        }

        let t = &*tensor;
        std::ptr::copy_nonoverlapping(data, t.data as *mut f32, count);

        tensor
    }
}

/// Create a tensor from a ZRTL array of f32
///
/// # Arguments
/// * `arr` - ZRTL array pointer (format: [i32 cap][i32 len][f32...])
///
/// # Returns
/// A 1D tensor containing a copy of the array data
#[no_mangle]
pub extern "C" fn tensor_from_array_f32(arr: zrtl::ArrayConstPtr) -> TensorPtr {
    if arr.is_null() {
        return TENSOR_NULL;
    }

    unsafe {
        let len = zrtl::array_length(arr) as usize;
        if len == 0 {
            return TENSOR_NULL;
        }

        let data_ptr: *const f32 = zrtl::array_data(arr);
        if data_ptr.is_null() {
            return TENSOR_NULL;
        }

        tensor_from_raw_f32(data_ptr, len)
    }
}

/// Create a tensor with values from 0 to n-1
#[no_mangle]
pub extern "C" fn tensor_arange_f32(start: f32, end: f32, step: f32) -> TensorPtr {
    if step == 0.0 || (end - start) / step < 0.0 {
        return TENSOR_NULL;
    }

    let n = ((end - start) / step).ceil() as usize;
    if n == 0 {
        return TENSOR_NULL;
    }

    let shape = [n];
    let tensor = tensor_new(shape.as_ptr(), 1, DType::F32 as u8);
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    unsafe {
        let t = &*tensor;
        let ptr = t.data as *mut f32;
        for i in 0..n {
            *ptr.add(i) = start + (i as f32) * step;
        }
    }

    tensor
}

/// Create a tensor with values in range (f64 wrapper for f32 version)
/// Accepts f64 parameters but creates f32 tensors (common ML use case)
#[no_mangle]
pub extern "C" fn tensor_arange(start: f64, end: f64, step: f64) -> TensorPtr {
    tensor_arange_f32(start as f32, end as f32, step as f32)
}

/// Create a tensor with n evenly spaced values between start and end
#[no_mangle]
pub extern "C" fn tensor_linspace_f32(start: f32, end: f32, n: usize) -> TensorPtr {
    if n == 0 {
        return TENSOR_NULL;
    }

    let shape = [n];
    let tensor = tensor_new(shape.as_ptr(), 1, DType::F32 as u8);
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    unsafe {
        let t = &*tensor;
        let ptr = t.data as *mut f32;
        if n == 1 {
            *ptr = start;
        } else {
            let step = (end - start) / (n - 1) as f32;
            for i in 0..n {
                *ptr.add(i) = start + (i as f32) * step;
            }
        }
    }

    tensor
}

/// Create a tensor with random values from uniform distribution [0, 1) (auto-seeded)
#[no_mangle]
pub extern "C" fn tensor_rand_f32_auto(
    shape_ptr: *const usize,
    ndim: u32,
) -> TensorPtr {
    tensor_rand_f32(shape_ptr, ndim, 0)
}

/// Create a tensor with random values from standard normal distribution (auto-seeded)
#[no_mangle]
pub extern "C" fn tensor_randn_f32_auto(
    shape_ptr: *const usize,
    ndim: u32,
) -> TensorPtr {
    tensor_randn_f32(shape_ptr, ndim, 0)
}

/// Create a tensor with random values from uniform distribution [0, 1)
#[no_mangle]
pub extern "C" fn tensor_rand_f32(
    shape_ptr: *const usize,
    ndim: u32,
    seed: u64,
) -> TensorPtr {
    let tensor = tensor_new(shape_ptr, ndim, DType::F32 as u8);
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    // Simple xorshift64 PRNG
    let mut state = if seed == 0 { 0x853c49e6748fea9b } else { seed };

    unsafe {
        let t = &*tensor;
        let ptr = t.data as *mut f32;
        let numel = t.numel();

        for i in 0..numel {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Convert to [0, 1)
            let val = (state & 0x7FFFFF) as f32 / (0x800000 as f32);
            *ptr.add(i) = val;
        }
    }

    tensor
}

/// Create a tensor with random values from standard normal distribution
#[no_mangle]
pub extern "C" fn tensor_randn_f32(
    shape_ptr: *const usize,
    ndim: u32,
    seed: u64,
) -> TensorPtr {
    let tensor = tensor_new(shape_ptr, ndim, DType::F32 as u8);
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    // Simple xorshift64 PRNG
    let mut state = if seed == 0 { 0x853c49e6748fea9b } else { seed };

    unsafe {
        let t = &*tensor;
        let ptr = t.data as *mut f32;
        let numel = t.numel();

        // Box-Muller transform for normal distribution
        let mut i = 0;
        while i < numel {
            // Generate two uniform random numbers
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u1 = ((state & 0x7FFFFF) as f32 / (0x800000 as f32)).max(1e-10);

            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u2 = (state & 0x7FFFFF) as f32 / (0x800000 as f32);

            // Box-Muller transform
            let mag = (-2.0 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
            let z1 = mag * (2.0 * std::f32::consts::PI * u2).sin();

            *ptr.add(i) = z0;
            if i + 1 < numel {
                *ptr.add(i + 1) = z1;
            }
            i += 2;
        }
    }

    tensor
}

// ============================================================================
// Memory Management
// ============================================================================

/// Free a tensor and its resources
#[no_mangle]
pub extern "C" fn tensor_free(tensor: TensorPtr) {
    if tensor.is_null() {
        return;
    }

    unsafe {
        let t = Box::from_raw(tensor);
        TensorStorage::decrement_refcount(t.storage);
    }
}

/// Clone a tensor (creates a new tensor with copied data)
#[no_mangle]
pub extern "C" fn tensor_clone(tensor: TensorPtr) -> TensorPtr {
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    unsafe {
        let t = &*tensor;
        let new_tensor = tensor_new(t.shape.as_ptr(), t.ndim, t.dtype as u8);
        if new_tensor.is_null() {
            return TENSOR_NULL;
        }

        let new_t = &mut *new_tensor;

        if t.is_contiguous() {
            // Fast path: copy contiguous memory
            ptr::copy_nonoverlapping(t.data, new_t.data, t.size_bytes());
        } else {
            // Slow path: iterate through all elements
            let elem_size = t.dtype.size_bytes();
            let numel = t.numel();
            let shape = &t.shape[..t.ndim as usize];

            // Use indices to iterate
            let mut indices = vec![0usize; t.ndim as usize];
            for flat in 0..numel {
                // Calculate source offset using strides
                let mut src_offset = 0isize;
                for d in 0..t.ndim as usize {
                    src_offset += (indices[d] as isize) * t.strides[d];
                }

                // Copy element
                ptr::copy_nonoverlapping(
                    t.data.offset(src_offset * elem_size as isize),
                    new_t.data.add(flat * elem_size),
                    elem_size,
                );

                // Increment indices
                for d in (0..t.ndim as usize).rev() {
                    indices[d] += 1;
                    if indices[d] < shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
        }

        new_tensor
    }
}

/// Create a view of a tensor (shares data, increments refcount)
#[no_mangle]
pub extern "C" fn tensor_view(tensor: TensorPtr) -> TensorPtr {
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    unsafe {
        let t = &*tensor;
        TensorStorage::increment_refcount(t.storage);

        let handle = Box::new(TensorHandle {
            storage: t.storage,
            data: t.data,
            shape: t.shape,
            strides: t.strides,
            ndim: t.ndim,
            dtype: t.dtype,
            layout: t.layout,
            flags: t.flags & !FLAG_OWNS_DATA,
        });

        Box::into_raw(handle)
    }
}

// ============================================================================
// Info Functions
// ============================================================================

/// Get number of dimensions
#[no_mangle]
pub extern "C" fn tensor_ndim(tensor: TensorPtr) -> u32 {
    if tensor.is_null() {
        return 0;
    }
    unsafe { (*tensor).ndim }
}

/// Get shape at dimension d
#[no_mangle]
pub extern "C" fn tensor_shape(tensor: TensorPtr, dim: u32) -> usize {
    if tensor.is_null() {
        return 0;
    }
    unsafe {
        let t = &*tensor;
        if dim >= t.ndim {
            return 0;
        }
        t.shape[dim as usize]
    }
}

/// Get stride at dimension d
#[no_mangle]
pub extern "C" fn tensor_stride(tensor: TensorPtr, dim: u32) -> isize {
    if tensor.is_null() {
        return 0;
    }
    unsafe {
        let t = &*tensor;
        if dim >= t.ndim {
            return 0;
        }
        t.strides[dim as usize]
    }
}

/// Get total number of elements
#[no_mangle]
pub extern "C" fn tensor_numel(tensor: TensorPtr) -> usize {
    if tensor.is_null() {
        return 0;
    }
    unsafe { (*tensor).numel() }
}

/// Get data type
#[no_mangle]
pub extern "C" fn tensor_dtype(tensor: TensorPtr) -> u8 {
    if tensor.is_null() {
        return 0;
    }
    unsafe { (*tensor).dtype as u8 }
}

/// Get raw data pointer
#[no_mangle]
pub extern "C" fn tensor_data(tensor: TensorPtr) -> *mut u8 {
    if tensor.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (*tensor).data }
}

/// Check if tensor is contiguous
#[no_mangle]
pub extern "C" fn tensor_is_contiguous(tensor: TensorPtr) -> bool {
    if tensor.is_null() {
        return false;
    }
    unsafe { (*tensor).is_contiguous() }
}

// ============================================================================
// Element Access
// ============================================================================

/// Get f32 element at flat index
#[no_mangle]
pub extern "C" fn tensor_get_f32(tensor: TensorPtr, index: usize) -> f32 {
    if tensor.is_null() {
        return 0.0;
    }
    unsafe {
        let t = &*tensor;
        if t.dtype != DType::F32 || index >= t.numel() {
            return 0.0;
        }
        *((t.data as *const f32).add(index))
    }
}

/// Set f32 element at flat index
#[no_mangle]
pub extern "C" fn tensor_set_f32(tensor: TensorPtr, index: usize, value: f32) {
    if tensor.is_null() {
        return;
    }
    unsafe {
        let t = &*tensor;
        if t.dtype != DType::F32 || index >= t.numel() || !t.is_writeable() {
            return;
        }
        *((t.data as *mut f32).add(index)) = value;
    }
}

/// Get element at multi-dimensional index for f32 tensor
#[no_mangle]
pub extern "C" fn tensor_get_at_f32(tensor: TensorPtr, indices: *const usize) -> f32 {
    if tensor.is_null() || indices.is_null() {
        return 0.0;
    }
    unsafe {
        let t = &*tensor;
        if t.dtype != DType::F32 {
            return 0.0;
        }

        // Calculate offset using strides
        let mut offset = 0isize;
        for d in 0..t.ndim as usize {
            let idx = *indices.add(d);
            if idx >= t.shape[d] {
                return 0.0;
            }
            offset += (idx as isize) * t.strides[d];
        }

        *((t.data as *const f32).offset(offset))
    }
}

/// Set element at multi-dimensional index for f32 tensor
#[no_mangle]
pub extern "C" fn tensor_set_at_f32(tensor: TensorPtr, indices: *const usize, value: f32) {
    if tensor.is_null() || indices.is_null() {
        return;
    }
    unsafe {
        let t = &*tensor;
        if t.dtype != DType::F32 || !t.is_writeable() {
            return;
        }

        // Calculate offset using strides
        let mut offset = 0isize;
        for d in 0..t.ndim as usize {
            let idx = *indices.add(d);
            if idx >= t.shape[d] {
                return;
            }
            offset += (idx as isize) * t.strides[d];
        }

        *((t.data as *mut f32).offset(offset)) = value;
    }
}

// ============================================================================
// Shape Operations
// ============================================================================

/// Reshape tensor (returns view if possible, clone if needed)
#[no_mangle]
pub extern "C" fn tensor_reshape(
    tensor: TensorPtr,
    new_shape: *const usize,
    new_ndim: u32,
) -> TensorPtr {
    if tensor.is_null() || new_shape.is_null() || new_ndim == 0 || new_ndim as usize > MAX_DIMS {
        return TENSOR_NULL;
    }

    unsafe {
        let t = &*tensor;

        // Calculate new total elements
        let mut new_numel = 1usize;
        let mut shape = [0usize; MAX_DIMS];
        for i in 0..new_ndim as usize {
            shape[i] = *new_shape.add(i);
            new_numel = new_numel.saturating_mul(shape[i]);
        }

        // Check that total elements match
        if new_numel != t.numel() {
            return TENSOR_NULL;
        }

        if t.is_contiguous() {
            // Create a view with new shape
            TensorStorage::increment_refcount(t.storage);

            let strides = TensorHandle::calculate_strides(&shape[..new_ndim as usize], t.layout);

            let handle = Box::new(TensorHandle {
                storage: t.storage,
                data: t.data,
                shape,
                strides,
                ndim: new_ndim,
                dtype: t.dtype,
                layout: t.layout,
                flags: t.flags & !FLAG_OWNS_DATA,
            });

            Box::into_raw(handle)
        } else {
            // Need to make contiguous first
            let contiguous = tensor_clone(tensor);
            if contiguous.is_null() {
                return TENSOR_NULL;
            }

            // Reshape the contiguous clone
            let c = &mut *contiguous;
            c.shape = shape;
            c.strides = TensorHandle::calculate_strides(&shape[..new_ndim as usize], c.layout);
            c.ndim = new_ndim;

            contiguous
        }
    }
}

/// Transpose tensor (swap two dimensions)
#[no_mangle]
pub extern "C" fn tensor_transpose(tensor: TensorPtr, dim0: u32, dim1: u32) -> TensorPtr {
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    unsafe {
        let t = &*tensor;
        if dim0 >= t.ndim || dim1 >= t.ndim {
            return TENSOR_NULL;
        }

        TensorStorage::increment_refcount(t.storage);

        let mut shape = t.shape;
        let mut strides = t.strides;

        // Swap dimensions
        shape.swap(dim0 as usize, dim1 as usize);
        strides.swap(dim0 as usize, dim1 as usize);

        let is_contiguous = TensorHandle::check_contiguous(
            &shape[..t.ndim as usize],
            &strides[..t.ndim as usize],
            t.layout,
        );

        let handle = Box::new(TensorHandle {
            storage: t.storage,
            data: t.data,
            shape,
            strides,
            ndim: t.ndim,
            dtype: t.dtype,
            layout: t.layout,
            flags: if is_contiguous {
                (t.flags | FLAG_CONTIGUOUS) & !FLAG_OWNS_DATA
            } else {
                (t.flags & !FLAG_CONTIGUOUS) & !FLAG_OWNS_DATA
            },
        });

        Box::into_raw(handle)
    }
}

/// Squeeze - remove dimensions of size 1
#[no_mangle]
pub extern "C" fn tensor_squeeze(tensor: TensorPtr) -> TensorPtr {
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    unsafe {
        let t = &*tensor;
        TensorStorage::increment_refcount(t.storage);

        let mut new_shape = [0usize; MAX_DIMS];
        let mut new_strides = [0isize; MAX_DIMS];
        let mut new_ndim = 0u32;

        for d in 0..t.ndim as usize {
            if t.shape[d] != 1 {
                new_shape[new_ndim as usize] = t.shape[d];
                new_strides[new_ndim as usize] = t.strides[d];
                new_ndim += 1;
            }
        }

        // Handle scalar case
        if new_ndim == 0 {
            new_ndim = 1;
            new_shape[0] = 1;
            new_strides[0] = 1;
        }

        let handle = Box::new(TensorHandle {
            storage: t.storage,
            data: t.data,
            shape: new_shape,
            strides: new_strides,
            ndim: new_ndim,
            dtype: t.dtype,
            layout: t.layout,
            flags: t.flags & !FLAG_OWNS_DATA,
        });

        Box::into_raw(handle)
    }
}

/// Unsqueeze - add dimension of size 1 at position
#[no_mangle]
pub extern "C" fn tensor_unsqueeze(tensor: TensorPtr, dim: u32) -> TensorPtr {
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    unsafe {
        let t = &*tensor;
        if dim > t.ndim || t.ndim as usize >= MAX_DIMS {
            return TENSOR_NULL;
        }

        TensorStorage::increment_refcount(t.storage);

        let mut new_shape = [0usize; MAX_DIMS];
        let mut new_strides = [0isize; MAX_DIMS];
        let new_ndim = t.ndim + 1;

        // Copy dimensions before insertion point
        for d in 0..dim as usize {
            new_shape[d] = t.shape[d];
            new_strides[d] = t.strides[d];
        }

        // Insert new dimension
        new_shape[dim as usize] = 1;
        // Stride for size-1 dimension doesn't matter, use next dim's stride
        new_strides[dim as usize] = if dim < t.ndim {
            t.strides[dim as usize]
        } else if t.ndim > 0 {
            1
        } else {
            1
        };

        // Copy dimensions after insertion point
        for d in dim as usize..t.ndim as usize {
            new_shape[d + 1] = t.shape[d];
            new_strides[d + 1] = t.strides[d];
        }

        let handle = Box::new(TensorHandle {
            storage: t.storage,
            data: t.data,
            shape: new_shape,
            strides: new_strides,
            ndim: new_ndim,
            dtype: t.dtype,
            layout: t.layout,
            flags: t.flags & !FLAG_OWNS_DATA,
        });

        Box::into_raw(handle)
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Sum all elements
#[no_mangle]
pub extern "C" fn tensor_sum_f32(tensor: TensorPtr) -> f32 {
    if tensor.is_null() {
        return 0.0;
    }

    unsafe {
        let t = &*tensor;
        if t.dtype != DType::F32 {
            return 0.0;
        }

        let numel = t.numel();

        if t.is_contiguous() {
            // Use SIMD-optimized sum for contiguous tensors
            vec_sum_f32(t.data as *const f32, numel as u64)
        } else {
            // Iterate using indices for non-contiguous tensors
            let shape = &t.shape[..t.ndim as usize];
            let mut indices = vec![0usize; t.ndim as usize];
            let mut sum = 0.0f32;

            for _ in 0..numel {
                let mut offset = 0isize;
                for d in 0..t.ndim as usize {
                    offset += (indices[d] as isize) * t.strides[d];
                }
                sum += *((t.data as *const f32).offset(offset));

                // Increment indices
                for d in (0..t.ndim as usize).rev() {
                    indices[d] += 1;
                    if indices[d] < shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            sum
        }
    }
}

/// Mean of all elements
#[no_mangle]
pub extern "C" fn tensor_mean_f32(tensor: TensorPtr) -> f32 {
    if tensor.is_null() {
        return 0.0;
    }

    unsafe {
        let t = &*tensor;
        let numel = t.numel();
        if numel == 0 {
            return 0.0;
        }
        tensor_sum_f32(tensor) / numel as f32
    }
}

/// Maximum value
#[no_mangle]
pub extern "C" fn tensor_max_f32(tensor: TensorPtr) -> f32 {
    if tensor.is_null() {
        return f32::NEG_INFINITY;
    }

    unsafe {
        let t = &*tensor;
        if t.dtype != DType::F32 {
            return f32::NEG_INFINITY;
        }

        let numel = t.numel();
        if numel == 0 {
            return f32::NEG_INFINITY;
        }

        if t.is_contiguous() {
            // Use SIMD-optimized max for contiguous tensors
            vec_max_f32(t.data as *const f32, numel as u64)
        } else {
            let shape = &t.shape[..t.ndim as usize];
            let mut indices = vec![0usize; t.ndim as usize];
            let mut max = f32::NEG_INFINITY;

            for _ in 0..numel {
                let mut offset = 0isize;
                for d in 0..t.ndim as usize {
                    offset += (indices[d] as isize) * t.strides[d];
                }
                let val = *((t.data as *const f32).offset(offset));
                if val > max {
                    max = val;
                }

                for d in (0..t.ndim as usize).rev() {
                    indices[d] += 1;
                    if indices[d] < shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            max
        }
    }
}

/// Minimum value
#[no_mangle]
pub extern "C" fn tensor_min_f32(tensor: TensorPtr) -> f32 {
    if tensor.is_null() {
        return f32::INFINITY;
    }

    unsafe {
        let t = &*tensor;
        if t.dtype != DType::F32 {
            return f32::INFINITY;
        }

        let numel = t.numel();
        if numel == 0 {
            return f32::INFINITY;
        }

        if t.is_contiguous() {
            // Use SIMD-optimized min for contiguous tensors
            vec_min_f32(t.data as *const f32, numel as u64)
        } else {
            let shape = &t.shape[..t.ndim as usize];
            let mut indices = vec![0usize; t.ndim as usize];
            let mut min = f32::INFINITY;

            for _ in 0..numel {
                let mut offset = 0isize;
                for d in 0..t.ndim as usize {
                    offset += (indices[d] as isize) * t.strides[d];
                }
                let val = *((t.data as *const f32).offset(offset));
                if val < min {
                    min = val;
                }

                for d in (0..t.ndim as usize).rev() {
                    indices[d] += 1;
                    if indices[d] < shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            min
        }
    }
}

/// Standard deviation of all elements
#[no_mangle]
pub extern "C" fn tensor_std(tensor: TensorPtr) -> f32 {
    if tensor.is_null() {
        return 0.0;
    }

    unsafe {
        let t = &*tensor;
        let numel = t.numel();
        if numel <= 1 {
            return 0.0;
        }
        tensor_var(tensor).sqrt()
    }
}

/// Variance of all elements
#[no_mangle]
pub extern "C" fn tensor_var(tensor: TensorPtr) -> f32 {
    if tensor.is_null() {
        return 0.0;
    }

    unsafe {
        let t = &*tensor;
        let numel = t.numel();
        if numel <= 1 {
            return 0.0;
        }
        let mean = tensor_mean_f32(tensor);

        if t.is_contiguous() {
            let data = t.data as *const f32;
            let mut sum_sq = 0.0f32;
            for i in 0..numel {
                let diff = *data.add(i) - mean;
                sum_sq += diff * diff;
            }
            sum_sq / (numel - 1) as f32
        } else {
            let shape = &t.shape[..t.ndim as usize];
            let mut indices = vec![0usize; t.ndim as usize];
            let mut sum_sq = 0.0f32;

            for _ in 0..numel {
                let mut offset = 0isize;
                for d in 0..t.ndim as usize {
                    offset += (indices[d] as isize) * t.strides[d];
                }
                let val = *((t.data as *const f32).offset(offset));
                let diff = val - mean;
                sum_sq += diff * diff;

                for d in (0..t.ndim as usize).rev() {
                    indices[d] += 1;
                    if indices[d] < shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            sum_sq / (numel - 1) as f32
        }
    }
}

/// Argmax - index of maximum value
#[no_mangle]
pub extern "C" fn tensor_argmax_f32(tensor: TensorPtr) -> usize {
    if tensor.is_null() {
        return 0;
    }

    unsafe {
        let t = &*tensor;
        if t.dtype != DType::F32 {
            return 0;
        }

        let numel = t.numel();
        if numel == 0 {
            return 0;
        }

        if t.is_contiguous() {
            // Use SIMD-optimized argmax for contiguous tensors
            let mut idx: u64 = 0;
            let mut val: f32 = 0.0;
            vec_argmax_with_val_f32(t.data as *const f32, numel as u64, &mut idx, &mut val);
            idx as usize
        } else {
            let shape = &t.shape[..t.ndim as usize];
            let mut indices = vec![0usize; t.ndim as usize];
            let mut max = f32::NEG_INFINITY;
            let mut max_idx = 0;

            for flat in 0..numel {
                let mut offset = 0isize;
                for d in 0..t.ndim as usize {
                    offset += (indices[d] as isize) * t.strides[d];
                }
                let val = *((t.data as *const f32).offset(offset));
                if val > max {
                    max = val;
                    max_idx = flat;
                }

                for d in (0..t.ndim as usize).rev() {
                    indices[d] += 1;
                    if indices[d] < shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            max_idx
        }
    }
}

// ============================================================================
// Slicing
// ============================================================================

/// Slice tensor along dimension
#[no_mangle]
pub extern "C" fn tensor_slice(
    tensor: TensorPtr,
    dim: u32,
    start: usize,
    end: usize,
) -> TensorPtr {
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    unsafe {
        let t = &*tensor;
        if dim >= t.ndim || start >= end || end > t.shape[dim as usize] {
            return TENSOR_NULL;
        }

        TensorStorage::increment_refcount(t.storage);

        let mut new_shape = t.shape;
        new_shape[dim as usize] = end - start;

        // Calculate new data pointer offset
        let offset_bytes =
            (start as isize) * t.strides[dim as usize] * (t.dtype.size_bytes() as isize);
        let new_data = t.data.offset(offset_bytes);

        let is_contiguous = TensorHandle::check_contiguous(
            &new_shape[..t.ndim as usize],
            &t.strides[..t.ndim as usize],
            t.layout,
        );

        let handle = Box::new(TensorHandle {
            storage: t.storage,
            data: new_data,
            shape: new_shape,
            strides: t.strides,
            ndim: t.ndim,
            dtype: t.dtype,
            layout: t.layout,
            flags: if is_contiguous {
                (t.flags | FLAG_CONTIGUOUS) & !FLAG_OWNS_DATA
            } else {
                (t.flags & !FLAG_CONTIGUOUS) & !FLAG_OWNS_DATA
            },
        });

        Box::into_raw(handle)
    }
}

// ============================================================================
// Type Conversion
// ============================================================================

/// Convert tensor to different dtype
#[no_mangle]
pub extern "C" fn tensor_to_dtype(tensor: TensorPtr, new_dtype: u8) -> TensorPtr {
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    let dtype = match new_dtype {
        0 => DType::F32,
        1 => DType::F64,
        4 => DType::I32,
        5 => DType::I64,
        _ => return TENSOR_NULL,
    };

    unsafe {
        let t = &*tensor;
        if t.dtype as u8 == new_dtype {
            // Same dtype, just return a view
            return tensor_view(tensor);
        }

        let new_tensor = tensor_new(t.shape.as_ptr(), t.ndim, new_dtype);
        if new_tensor.is_null() {
            return TENSOR_NULL;
        }

        let new_t = &*new_tensor;
        let numel = t.numel();

        // Convert each element
        for i in 0..numel {
            let src_val: f64 = match t.dtype {
                DType::F32 => *((t.data as *const f32).add(i)) as f64,
                DType::F64 => *((t.data as *const f64).add(i)),
                DType::I32 => *((t.data as *const i32).add(i)) as f64,
                DType::I64 => *((t.data as *const i64).add(i)) as f64,
                _ => 0.0,
            };

            match dtype {
                DType::F32 => *((new_t.data as *mut f32).add(i)) = src_val as f32,
                DType::F64 => *((new_t.data as *mut f64).add(i)) = src_val,
                DType::I32 => *((new_t.data as *mut i32).add(i)) = src_val as i32,
                DType::I64 => *((new_t.data as *mut i64).add(i)) = src_val as i64,
                _ => {}
            }
        }

        new_tensor
    }
}

/// Make tensor contiguous (copy if needed)
#[no_mangle]
pub extern "C" fn tensor_contiguous(tensor: TensorPtr) -> TensorPtr {
    if tensor.is_null() {
        return TENSOR_NULL;
    }

    unsafe {
        let t = &*tensor;
        if t.is_contiguous() {
            return tensor_view(tensor);
        }

        tensor_clone(tensor)
    }
}

// ============================================================================
// Display / Printing Functions
// ============================================================================

/// Print tensor contents to stdout
/// Format: tensor([1.0, 2.0, 3.0], shape=[3], dtype=f32)
#[no_mangle]
pub extern "C" fn tensor_print(tensor: TensorPtr) {
    use std::io::Write;

    if tensor.is_null() {
        print!("tensor(null)");
        return;
    }

    unsafe {
        let t = &*tensor;
        let numel = t.numel();
        let shape = &t.shape[..t.ndim as usize];

        print!("tensor([");

        // Print elements (limit to first 10 for readability)
        let max_print = 10.min(numel);
        for i in 0..max_print {
            if i > 0 {
                print!(", ");
            }

            // Get element value based on dtype
            match t.dtype {
                DType::F32 => {
                    let val = *(t.data as *const f32).add(i);
                    if val.fract() == 0.0 && val.abs() < 1e6 {
                        print!("{:.1}", val);
                    } else {
                        print!("{:.6}", val);
                    }
                }
                DType::F64 => {
                    let val = *(t.data as *const f64).add(i);
                    if val.fract() == 0.0 && val.abs() < 1e6 {
                        print!("{:.1}", val);
                    } else {
                        print!("{:.6}", val);
                    }
                }
                DType::I32 => print!("{}", *(t.data as *const i32).add(i)),
                DType::I64 => print!("{}", *(t.data as *const i64).add(i)),
                _ => print!("?"),
            }
        }

        if numel > max_print {
            print!(", ... ({} more)", numel - max_print);
        }

        print!("], shape=[");
        for (i, &dim) in shape.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{}", dim);
        }
        print!("]");

        // Print dtype
        match t.dtype {
            DType::F32 => print!(", dtype=f32"),
            DType::F64 => print!(", dtype=f64"),
            DType::I32 => print!(", dtype=i32"),
            DType::I64 => print!(", dtype=i64"),
            _ => print!(", dtype=?"),
        }

        print!(")");
        std::io::stdout().flush().ok();
    }
}

/// Print tensor contents to stdout with newline
#[no_mangle]
pub extern "C" fn tensor_println(tensor: TensorPtr) {
    tensor_print(tensor);
    println!();
}

/// Convert tensor to string representation (for Display trait)
/// Returns a ZRTL string (heap-allocated, caller takes ownership)
/// Format: tensor([1.0, 2.0, 3.0], shape=[3], dtype=f32)
#[no_mangle]
pub extern "C" fn tensor_to_string(tensor: TensorPtr) -> *mut u8 {
    let result = if tensor.is_null() {
        "tensor(null)".to_string()
    } else {
        unsafe {
            let t = &*tensor;
            let numel = t.numel();
            let shape = &t.shape[..t.ndim as usize];

            let mut s = String::with_capacity(128);
            s.push_str("tensor([");

            // Format elements (limit to first 10 for readability)
            let max_print = 10.min(numel);
            for i in 0..max_print {
                if i > 0 {
                    s.push_str(", ");
                }

                // Get element value based on dtype
                match t.dtype {
                    DType::F32 => {
                        let val = *(t.data as *const f32).add(i);
                        if val.fract() == 0.0 && val.abs() < 1e6 {
                            s.push_str(&format!("{:.1}", val));
                        } else {
                            s.push_str(&format!("{:.6}", val));
                        }
                    }
                    DType::F64 => {
                        let val = *(t.data as *const f64).add(i);
                        if val.fract() == 0.0 && val.abs() < 1e6 {
                            s.push_str(&format!("{:.1}", val));
                        } else {
                            s.push_str(&format!("{:.6}", val));
                        }
                    }
                    DType::I32 => s.push_str(&format!("{}", *(t.data as *const i32).add(i))),
                    DType::I64 => s.push_str(&format!("{}", *(t.data as *const i64).add(i))),
                    _ => s.push('?'),
                }
            }

            if numel > max_print {
                s.push_str(&format!(", ... ({} more)", numel - max_print));
            }

            s.push_str("], shape=[");
            for (i, &dim) in shape.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&format!("{}", dim));
            }
            s.push(']');

            // Add dtype
            match t.dtype {
                DType::F32 => s.push_str(", dtype=f32"),
                DType::F64 => s.push_str(", dtype=f64"),
                DType::I32 => s.push_str(", dtype=i32"),
                DType::I64 => s.push_str(", dtype=i64"),
                _ => s.push_str(", dtype=?"),
            }

            s.push(')');
            s
        }
    };

    // Convert to ZRTL string (heap-allocated)
    // ZRTL string format: [i32 length][utf8 bytes...]
    // Header size is 4 bytes (one i32 for length)
    let bytes = result.as_bytes();
    let len = bytes.len();

    // Allocate: [i32 len][bytes...]
    let total_size = 4 + len;
    let layout = std::alloc::Layout::from_size_align(total_size, 4).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) };

    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        // Write length at offset 0
        *(ptr as *mut i32) = len as i32;
        // Copy string data at offset 4
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(4), len);
    }

    ptr
}

// ============================================================================
// Arithmetic Operators (for trait dispatch)
// ============================================================================

/// Element-wise addition: a + b
/// Broadcasts if shapes don't match
#[no_mangle]
pub extern "C" fn tensor_add(a: TensorPtr, b: TensorPtr) -> TensorPtr {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let ta = &*a;
        let tb = &*b;

        // Simple case: same shape
        if ta.shape[..ta.ndim as usize] == tb.shape[..tb.ndim as usize] {
            let result = tensor_clone(a);
            let tr = &mut *result;

            // F32 only for now
            if ta.dtype == DType::F32 && tb.dtype == DType::F32 {
                let numel = ta.numel();
                let ra = tr.data as *mut f32;
                let rb = tb.data as *const f32;
                for i in 0..numel {
                    *ra.add(i) += *rb.add(i);
                }
            }
            result
        } else {
            // Broadcasting not yet implemented - return clone of a
            tensor_clone(a)
        }
    }
}

/// Element-wise subtraction: a - b
#[no_mangle]
pub extern "C" fn tensor_sub(a: TensorPtr, b: TensorPtr) -> TensorPtr {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let ta = &*a;
        let tb = &*b;

        if ta.shape[..ta.ndim as usize] == tb.shape[..tb.ndim as usize] {
            let result = tensor_clone(a);
            let tr = &mut *result;

            if ta.dtype == DType::F32 && tb.dtype == DType::F32 {
                let numel = ta.numel();
                let ra = tr.data as *mut f32;
                let rb = tb.data as *const f32;
                for i in 0..numel {
                    *ra.add(i) -= *rb.add(i);
                }
            }
            result
        } else {
            tensor_clone(a)
        }
    }
}

/// Element-wise multiplication: a * b
#[no_mangle]
pub extern "C" fn tensor_mul(a: TensorPtr, b: TensorPtr) -> TensorPtr {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let ta = &*a;
        let tb = &*b;

        if ta.shape[..ta.ndim as usize] == tb.shape[..tb.ndim as usize] {
            let result = tensor_clone(a);
            let tr = &mut *result;

            if ta.dtype == DType::F32 && tb.dtype == DType::F32 {
                let numel = ta.numel();
                let ra = tr.data as *mut f32;
                let rb = tb.data as *const f32;
                for i in 0..numel {
                    *ra.add(i) *= *rb.add(i);
                }
            }
            result
        } else {
            tensor_clone(a)
        }
    }
}

/// Element-wise division: a / b
#[no_mangle]
pub extern "C" fn tensor_div(a: TensorPtr, b: TensorPtr) -> TensorPtr {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let ta = &*a;
        let tb = &*b;

        if ta.shape[..ta.ndim as usize] == tb.shape[..tb.ndim as usize] {
            let result = tensor_clone(a);
            let tr = &mut *result;

            if ta.dtype == DType::F32 && tb.dtype == DType::F32 {
                let numel = ta.numel();
                let ra = tr.data as *mut f32;
                let rb = tb.data as *const f32;
                for i in 0..numel {
                    let divisor = *rb.add(i);
                    if divisor != 0.0 {
                        *ra.add(i) /= divisor;
                    } else {
                        *ra.add(i) = f32::INFINITY;
                    }
                }
            }
            result
        } else {
            tensor_clone(a)
        }
    }
}

/// Unary negation: -a
#[no_mangle]
pub extern "C" fn tensor_neg(a: TensorPtr) -> TensorPtr {
    if a.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let ta = &*a;
        let result = tensor_clone(a);
        let tr = &mut *result;

        if ta.dtype == DType::F32 {
            let numel = ta.numel();
            let ra = tr.data as *mut f32;
            for i in 0..numel {
                *ra.add(i) = -*ra.add(i);
            }
        }
        result
    }
}

/// Dot product / matrix multiplication: a @ b
/// For 1D tensors: dot product (returns scalar)
/// For 2D tensors: matrix multiplication
#[no_mangle]
pub extern "C" fn tensor_dot(a: TensorPtr, b: TensorPtr) -> f32 {
    if a.is_null() || b.is_null() {
        return 0.0;
    }

    unsafe {
        let ta = &*a;
        let tb = &*b;

        // Only F32 supported
        if ta.dtype != DType::F32 || tb.dtype != DType::F32 {
            return 0.0;
        }

        // 1D dot product
        let numel_a = ta.numel();
        let numel_b = tb.numel();

        if numel_a != numel_b {
            return 0.0;
        }

        let pa = ta.data as *const f32;
        let pb = tb.data as *const f32;

        let mut sum = 0.0f32;
        for i in 0..numel_a {
            sum += (*pa.add(i)) * (*pb.add(i));
        }
        sum
    }
}

/// Element-wise modulo: a % b
#[no_mangle]
pub extern "C" fn tensor_mod(a: TensorPtr, b: TensorPtr) -> TensorPtr {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let ta = &*a;
        let tb = &*b;

        if ta.shape[..ta.ndim as usize] == tb.shape[..tb.ndim as usize] {
            let result = tensor_clone(a);
            let tr = &mut *result;

            if ta.dtype == DType::F32 && tb.dtype == DType::F32 {
                let numel = ta.numel();
                let ra = tr.data as *mut f32;
                let rb = tb.data as *const f32;
                for i in 0..numel {
                    let divisor = *rb.add(i);
                    if divisor != 0.0 {
                        *ra.add(i) %= divisor;
                    }
                }
            }
            result
        } else {
            tensor_clone(a)
        }
    }
}

// ============================================================================
// Element-wise Math Operations
// ============================================================================

/// Helper: apply a unary f32 operation element-wise, return new tensor
unsafe fn unary_elementwise_f32(tensor: TensorPtr, op: impl Fn(f32) -> f32) -> TensorPtr {
    if tensor.is_null() {
        return TENSOR_NULL;
    }
    let t = &*tensor;
    let result = tensor_clone(tensor);
    if result.is_null() {
        return TENSOR_NULL;
    }
    let tr = &mut *result;
    if t.dtype == DType::F32 {
        let numel = t.numel();
        let data = tr.data as *mut f32;
        for i in 0..numel {
            *data.add(i) = op(*data.add(i));
        }
    }
    result
}

/// Helper: apply a binary comparison on two tensors, return f32 tensor (1.0 = true, 0.0 = false)
unsafe fn binary_cmp_f32(a: TensorPtr, b: TensorPtr, cmp: impl Fn(f32, f32) -> bool) -> TensorPtr {
    if a.is_null() || b.is_null() {
        return TENSOR_NULL;
    }
    let ta = &*a;
    let tb = &*b;
    if ta.shape[..ta.ndim as usize] != tb.shape[..tb.ndim as usize] {
        return TENSOR_NULL;
    }
    let result = tensor_clone(a);
    if result.is_null() {
        return TENSOR_NULL;
    }
    let tr = &mut *result;
    if ta.dtype == DType::F32 && tb.dtype == DType::F32 {
        let numel = ta.numel();
        let ra = tr.data as *mut f32;
        let rb = tb.data as *const f32;
        for i in 0..numel {
            *ra.add(i) = if cmp(*ra.add(i), *rb.add(i)) { 1.0 } else { 0.0 };
        }
    }
    result
}

/// Helper: axis-wise reduction returning a new tensor
unsafe fn axis_reduce_f32(
    tensor: TensorPtr,
    axis: u32,
    reduce: impl Fn(&[f32]) -> f32,
) -> TensorPtr {
    if tensor.is_null() {
        return TENSOR_NULL;
    }
    let t = &*tensor;
    if axis >= t.ndim {
        return TENSOR_NULL;
    }
    let ndim = t.ndim as usize;
    let axis_size = t.shape[axis as usize];
    if axis_size == 0 {
        return TENSOR_NULL;
    }

    // Build output shape: remove the axis dimension
    let mut out_shape = [0usize; 8];
    let mut out_ndim = 0usize;
    for d in 0..ndim {
        if d != axis as usize {
            out_shape[out_ndim] = t.shape[d];
            out_ndim += 1;
        }
    }

    // Handle scalar output (reducing a 1-d tensor along axis 0)
    if out_ndim == 0 {
        out_shape[0] = 1;
        out_ndim = 1;
    }

    let result = tensor_new(out_shape.as_ptr(), out_ndim as u32, DType::F32 as u8);
    if result.is_null() {
        return TENSOR_NULL;
    }
    let tr = &*result;
    let out_numel = tr.numel();
    let out_data = tr.data as *mut f32;

    // Iterate over all positions in the output tensor
    let mut out_indices = vec![0usize; out_ndim];
    let mut buf = vec![0.0f32; axis_size];

    for out_flat in 0..out_numel {
        // Map output indices back to input indices (inserting axis dimension)
        // Gather along the axis
        for a_i in 0..axis_size {
            let mut src_offset = 0isize;
            let mut out_d = 0usize;
            for d in 0..ndim {
                let idx = if d == axis as usize {
                    a_i
                } else {
                    let v = out_indices[out_d];
                    out_d += 1;
                    v
                };
                src_offset += (idx as isize) * t.strides[d];
            }
            buf[a_i] = *((t.data as *const f32).offset(src_offset));
        }

        *out_data.add(out_flat) = reduce(&buf);

        // Increment output indices
        for d in (0..out_ndim).rev() {
            out_indices[d] += 1;
            if out_indices[d] < out_shape[d] {
                break;
            }
            out_indices[d] = 0;
        }
    }

    result
}

#[no_mangle]
pub extern "C" fn tensor_abs(tensor: TensorPtr) -> TensorPtr {
    unsafe { unary_elementwise_f32(tensor, |x| x.abs()) }
}

#[no_mangle]
pub extern "C" fn tensor_sqrt(tensor: TensorPtr) -> TensorPtr {
    unsafe { unary_elementwise_f32(tensor, |x| x.sqrt()) }
}

#[no_mangle]
pub extern "C" fn tensor_exp(tensor: TensorPtr) -> TensorPtr {
    unsafe { unary_elementwise_f32(tensor, |x| x.exp()) }
}

#[no_mangle]
pub extern "C" fn tensor_log(tensor: TensorPtr) -> TensorPtr {
    unsafe { unary_elementwise_f32(tensor, |x| x.ln()) }
}

#[no_mangle]
pub extern "C" fn tensor_sin(tensor: TensorPtr) -> TensorPtr {
    unsafe { unary_elementwise_f32(tensor, |x| x.sin()) }
}

#[no_mangle]
pub extern "C" fn tensor_cos(tensor: TensorPtr) -> TensorPtr {
    unsafe { unary_elementwise_f32(tensor, |x| x.cos()) }
}

#[no_mangle]
pub extern "C" fn tensor_tanh(tensor: TensorPtr) -> TensorPtr {
    unsafe { unary_elementwise_f32(tensor, |x| x.tanh()) }
}

#[no_mangle]
pub extern "C" fn tensor_pow(tensor: TensorPtr, exponent: f64) -> TensorPtr {
    unsafe { unary_elementwise_f32(tensor, |x| x.powf(exponent as f32)) }
}

#[no_mangle]
pub extern "C" fn tensor_clamp(tensor: TensorPtr, min_val: f64, max_val: f64) -> TensorPtr {
    unsafe { unary_elementwise_f32(tensor, |x| x.max(min_val as f32).min(max_val as f32)) }
}

#[no_mangle]
pub extern "C" fn tensor_relu(tensor: TensorPtr) -> TensorPtr {
    unsafe { unary_elementwise_f32(tensor, |x| x.max(0.0)) }
}

#[no_mangle]
pub extern "C" fn tensor_sigmoid(tensor: TensorPtr) -> TensorPtr {
    unsafe { unary_elementwise_f32(tensor, |x| 1.0 / (1.0 + (-x).exp())) }
}

// ============================================================================
// Comparison Operations
// ============================================================================

#[no_mangle]
pub extern "C" fn tensor_eq(a: TensorPtr, b: TensorPtr) -> TensorPtr {
    unsafe { binary_cmp_f32(a, b, |x, y| (x - y).abs() < f32::EPSILON) }
}

#[no_mangle]
pub extern "C" fn tensor_ne(a: TensorPtr, b: TensorPtr) -> TensorPtr {
    unsafe { binary_cmp_f32(a, b, |x, y| (x - y).abs() >= f32::EPSILON) }
}

#[no_mangle]
pub extern "C" fn tensor_lt(a: TensorPtr, b: TensorPtr) -> TensorPtr {
    unsafe { binary_cmp_f32(a, b, |x, y| x < y) }
}

#[no_mangle]
pub extern "C" fn tensor_le(a: TensorPtr, b: TensorPtr) -> TensorPtr {
    unsafe { binary_cmp_f32(a, b, |x, y| x <= y) }
}

#[no_mangle]
pub extern "C" fn tensor_gt(a: TensorPtr, b: TensorPtr) -> TensorPtr {
    unsafe { binary_cmp_f32(a, b, |x, y| x > y) }
}

#[no_mangle]
pub extern "C" fn tensor_ge(a: TensorPtr, b: TensorPtr) -> TensorPtr {
    unsafe { binary_cmp_f32(a, b, |x, y| x >= y) }
}

// ============================================================================
// Axis-wise Reductions
// ============================================================================

#[no_mangle]
pub extern "C" fn tensor_sum_axis(tensor: TensorPtr, axis: u32) -> TensorPtr {
    unsafe {
        axis_reduce_f32(tensor, axis, |vals| vals.iter().sum())
    }
}

#[no_mangle]
pub extern "C" fn tensor_mean_axis(tensor: TensorPtr, axis: u32) -> TensorPtr {
    unsafe {
        axis_reduce_f32(tensor, axis, |vals| {
            if vals.is_empty() { 0.0 } else { vals.iter().sum::<f32>() / vals.len() as f32 }
        })
    }
}

#[no_mangle]
pub extern "C" fn tensor_max_axis(tensor: TensorPtr, axis: u32) -> TensorPtr {
    unsafe {
        axis_reduce_f32(tensor, axis, |vals| {
            vals.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        })
    }
}

#[no_mangle]
pub extern "C" fn tensor_min_axis(tensor: TensorPtr, axis: u32) -> TensorPtr {
    unsafe {
        axis_reduce_f32(tensor, axis, |vals| {
            vals.iter().copied().fold(f32::INFINITY, f32::min)
        })
    }
}

// ============================================================================
// Shape Operations
// ============================================================================

#[no_mangle]
pub extern "C" fn tensor_flatten(tensor: TensorPtr) -> TensorPtr {
    if tensor.is_null() {
        return TENSOR_NULL;
    }
    unsafe {
        let t = &*tensor;
        let numel = t.numel();
        let shape = [numel];
        tensor_reshape(tensor, shape.as_ptr(), 1)
    }
}

// ============================================================================
// Conversion and Utility
// ============================================================================

/// Convert tensor to list of f64 values
#[no_mangle]
pub extern "C" fn tensor_to_list(tensor: TensorPtr) -> *mut u8 {
    if tensor.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let t = &*tensor;
        let numel = t.numel();

        // Allocate a List<f64> struct: {data_ptr: i64, len: i64, capacity: i64}
        // Plus the data buffer itself
        let data_layout = std::alloc::Layout::from_size_align(numel * 8, 8).unwrap();
        let data_ptr = std::alloc::alloc(data_layout);
        if data_ptr.is_null() {
            return std::ptr::null_mut();
        }

        let out = data_ptr as *mut f64;
        for i in 0..numel {
            let val = match t.dtype {
                DType::F32 => *(t.data as *const f32).add(i) as f64,
                DType::F64 => *(t.data as *const f64).add(i),
                DType::I32 => *(t.data as *const i32).add(i) as f64,
                DType::I64 => *(t.data as *const i64).add(i) as f64,
                _ => 0.0,
            };
            *out.add(i) = val;
        }

        // Build List struct: {i64 data_ptr, i64 len, i64 capacity}
        let list_layout = std::alloc::Layout::from_size_align(24, 8).unwrap();
        let list_ptr = std::alloc::alloc(list_layout);
        if list_ptr.is_null() {
            std::alloc::dealloc(data_ptr, data_layout);
            return std::ptr::null_mut();
        }
        *(list_ptr as *mut i64) = data_ptr as i64;
        *((list_ptr as *mut i64).add(1)) = numel as i64;
        *((list_ptr as *mut i64).add(2)) = numel as i64;

        list_ptr
    }
}

/// Extract scalar value from single-element tensor
#[no_mangle]
pub extern "C" fn tensor_item(tensor: TensorPtr) -> f64 {
    if tensor.is_null() {
        return 0.0;
    }
    unsafe {
        let t = &*tensor;
        match t.dtype {
            DType::F32 => *(t.data as *const f32) as f64,
            DType::F64 => *(t.data as *const f64),
            DType::I32 => *(t.data as *const i32) as f64,
            DType::I64 => *(t.data as *const i64) as f64,
            _ => 0.0,
        }
    }
}

/// Softmax along axis
#[no_mangle]
pub extern "C" fn tensor_softmax(tensor: TensorPtr, axis: i64) -> TensorPtr {
    if tensor.is_null() {
        return TENSOR_NULL;
    }
    unsafe {
        let t = &*tensor;
        let ndim = t.ndim as usize;
        let actual_axis = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };
        if actual_axis >= ndim {
            return TENSOR_NULL;
        }

        // Simple case: 1D tensor
        let result = tensor_clone(tensor);
        if result.is_null() {
            return TENSOR_NULL;
        }

        if ndim == 1 && t.dtype == DType::F32 {
            let tr = &mut *result;
            let numel = t.numel();
            let data = tr.data as *mut f32;

            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..numel {
                if *data.add(i) > max_val {
                    max_val = *data.add(i);
                }
            }

            // exp(x - max) and sum
            let mut sum = 0.0f32;
            for i in 0..numel {
                let v = (*data.add(i) - max_val).exp();
                *data.add(i) = v;
                sum += v;
            }

            // Normalize
            if sum > 0.0 {
                for i in 0..numel {
                    *data.add(i) /= sum;
                }
            }
        }
        // For multi-dim, just return the clone (simplified)
        result
    }
}

/// Cross entropy loss
#[no_mangle]
pub extern "C" fn tensor_cross_entropy(pred: TensorPtr, target: TensorPtr) -> f64 {
    if pred.is_null() || target.is_null() {
        return 0.0;
    }
    unsafe {
        let tp = &*pred;
        let tt = &*target;
        let numel = tp.numel().min(tt.numel());
        if numel == 0 || tp.dtype != DType::F32 || tt.dtype != DType::F32 {
            return 0.0;
        }

        let pp = tp.data as *const f32;
        let pt = tt.data as *const f32;

        let mut loss = 0.0f64;
        for i in 0..numel {
            let p = (*pp.add(i)).max(1e-7) as f64; // clamp to avoid log(0)
            let t = *pt.add(i) as f64;
            loss -= t * p.ln();
        }
        loss / numel as f64
    }
}

/// Concatenate tensors along axis (simplified: takes two tensors)
#[no_mangle]
pub extern "C" fn tensor_concat(_tensors: *const TensorPtr, _axis: i64) -> TensorPtr {
    TENSOR_NULL
}

/// Stack tensors along new axis (stub)
#[no_mangle]
pub extern "C" fn tensor_stack(_tensors: *const TensorPtr, _axis: i64) -> TensorPtr {
    TENSOR_NULL
}

/// Split tensor into chunks (stub)
#[no_mangle]
pub extern "C" fn tensor_split(_tensor: TensorPtr, _chunks: i64, _axis: i64) -> *mut u8 {
    std::ptr::null_mut()
}

/// Transpose with axis swap (alias for the main transpose)
#[no_mangle]
pub extern "C" fn tensor_transpose_axes(tensor: TensorPtr, dim0: u32, dim1: u32) -> TensorPtr {
    tensor_transpose(tensor, dim0, dim1)
}

// ============================================================================
// Plugin Registration
// ============================================================================

zrtl_plugin! {
    name: "tensor",
    symbols: [
        // Creation - all return TensorPtr (opaque pointers)
        // Note: pointers are i64, counts/sizes are i64 or u64 depending on usize
        ("$Tensor$new", tensor_new, (i64, u32, u8) -> opaque),  // shape_ptr, ndim, dtype
        ("$Tensor$zeros", tensor_zeros_f32, (i64, u32) -> opaque),  // shape_ptr, ndim (f32 default)
        ("$Tensor$ones", tensor_ones_f32, (i64, u32) -> opaque),  // shape_ptr, ndim (f32 default)
        ("$Tensor$zeros_typed", tensor_zeros, (i64, u32, u8) -> opaque),  // shape_ptr, ndim, dtype
        ("$Tensor$ones_typed", tensor_ones, (i64, u32, u8) -> opaque),  // shape_ptr, ndim, dtype
        ("$Tensor$zeros_1d", tensor_zeros_1d, (i64) -> opaque),  // n
        ("$Tensor$zeros_2d", tensor_zeros_2d, (i64, i64) -> opaque),  // rows, cols
        ("$Tensor$zeros_3d", tensor_zeros_3d, (i64, i64, i64) -> opaque),  // d0, d1, d2
        ("$Tensor$ones_1d", tensor_ones_1d, (i64) -> opaque),  // n
        ("$Tensor$ones_2d", tensor_ones_2d, (i64, i64) -> opaque),  // rows, cols
        ("$Tensor$ones_3d", tensor_ones_3d, (i64, i64, i64) -> opaque),  // d0, d1, d2
        ("$Tensor$full_f32", tensor_full_f32, (i64, u32, f32) -> opaque),  // shape_ptr, ndim, value
        ("$Tensor$from_array_f32", tensor_from_array_f32, (i64) -> opaque),  // array ptr
        ("$Tensor$from_raw_f32", tensor_from_raw_f32, (i64, u64) -> opaque),  // ptr + count (usize)
        ("$Tensor$arange_f32", tensor_arange_f32, (f32, f32, f32) -> opaque),
        ("$Tensor$arange", tensor_arange, (f64, f64, f64) -> opaque),  // f64 wrapper for convenience
        ("$Tensor$linspace_f32", tensor_linspace_f32, (f32, f32, u64) -> opaque),  // start, end, n (usize)
        ("$Tensor$rand_f32", tensor_rand_f32_auto, (i64, u32) -> opaque),  // shape_ptr, ndim (auto-seeded)
        ("$Tensor$randn_f32", tensor_randn_f32_auto, (i64, u32) -> opaque),  // shape_ptr, ndim (auto-seeded)
        ("$Tensor$rand_f32_seeded", tensor_rand_f32, (i64, u32, u64) -> opaque),  // shape_ptr, ndim, seed
        ("$Tensor$randn_f32_seeded", tensor_randn_f32, (i64, u32, u64) -> opaque),  // shape_ptr, ndim, seed

        // Memory
        ("$Tensor$free", tensor_free, (i64) -> void),
        ("$Tensor$clone", tensor_clone, (i64) -> opaque),
        ("$Tensor$view", tensor_view, (i64) -> opaque),
        ("$Tensor$contiguous", tensor_contiguous, (i64) -> opaque),

        // Info
        ("$Tensor$ndim", tensor_ndim, (i64) -> u32),
        ("$Tensor$shape", tensor_shape, (i64, u32) -> u64),
        ("$Tensor$stride", tensor_stride, (i64, u32) -> i64),
        ("$Tensor$numel", tensor_numel, (i64) -> u64),
        ("$Tensor$dtype", tensor_dtype, (i64) -> u8),
        ("$Tensor$data", tensor_data, (i64) -> i64),
        ("$Tensor$is_contiguous", tensor_is_contiguous, (i64) -> bool),

        // Element access
        ("$Tensor$get_f32", tensor_get_f32, (i64, i64) -> f32),
        ("$Tensor$set_f32", tensor_set_f32, (i64, i64, f32) -> void),
        ("$Tensor$get_at_f32", tensor_get_at_f32, (i64, i64) -> f32),
        ("$Tensor$set_at_f32", tensor_set_at_f32, (i64, i64, f32) -> void),

        // Shape operations - return new tensors
        ("$Tensor$reshape", tensor_reshape, (i64, i64, u32) -> opaque),
        ("$Tensor$transpose", tensor_transpose, (i64, u32, u32) -> opaque),
        ("$Tensor$squeeze", tensor_squeeze, (i64) -> opaque),
        ("$Tensor$unsqueeze", tensor_unsqueeze, (i64, u32) -> opaque),
        ("$Tensor$slice", tensor_slice, (i64, u32, i64, i64) -> opaque),

        // Reductions - return scalars
        ("$Tensor$sum_f32", tensor_sum_f32, (i64) -> f32),
        ("$Tensor$mean_f32", tensor_mean_f32, (i64) -> f32),
        ("$Tensor$max_f32", tensor_max_f32, (i64) -> f32),
        ("$Tensor$min_f32", tensor_min_f32, (i64) -> f32),
        ("$Tensor$std_f32", tensor_std, (i64) -> f32),
        ("$Tensor$var_f32", tensor_var, (i64) -> f32),
        ("$Tensor$argmax_f32", tensor_argmax_f32, (i64) -> i64),

        // Type conversion
        ("$Tensor$to_dtype", tensor_to_dtype, (i64, u8) -> opaque),

        // Display trait implementation
        ("$Tensor$to_string", tensor_to_string, (i64) -> i64),
        ("$Tensor$print", tensor_print, (i64) -> void),
        ("$Tensor$println", tensor_println, (i64) -> void),

        // Arithmetic operator trait methods - return new tensors
        ("$Tensor$add", tensor_add, (i64, i64) -> opaque),
        ("$Tensor$sub", tensor_sub, (i64, i64) -> opaque),
        ("$Tensor$mul", tensor_mul, (i64, i64) -> opaque),
        ("$Tensor$div", tensor_div, (i64, i64) -> opaque),
        ("$Tensor$mod", tensor_mod, (i64, i64) -> opaque),
        ("$Tensor$neg", tensor_neg, (i64) -> opaque),
        ("$Tensor$matmul", tensor_dot, (i64, i64) -> f32),  // @ operator returns f32, not tensor
        ("$Tensor$dot", tensor_dot, (i64, i64) -> f32),

        // Element-wise math operations
        ("$Tensor$abs", tensor_abs, (i64) -> opaque),
        ("$Tensor$sqrt", tensor_sqrt, (i64) -> opaque),
        ("$Tensor$exp", tensor_exp, (i64) -> opaque),
        ("$Tensor$log", tensor_log, (i64) -> opaque),
        ("$Tensor$sin", tensor_sin, (i64) -> opaque),
        ("$Tensor$cos", tensor_cos, (i64) -> opaque),
        ("$Tensor$tanh", tensor_tanh, (i64) -> opaque),
        ("$Tensor$pow", tensor_pow, (i64, f64) -> opaque),
        ("$Tensor$clamp", tensor_clamp, (i64, f64, f64) -> opaque),
        ("$Tensor$relu", tensor_relu, (i64) -> opaque),
        ("$Tensor$sigmoid", tensor_sigmoid, (i64) -> opaque),

        // Comparison operations (return f32 boolean tensor)
        ("$Tensor$eq", tensor_eq, (i64, i64) -> opaque),
        ("$Tensor$ne", tensor_ne, (i64, i64) -> opaque),
        ("$Tensor$lt", tensor_lt, (i64, i64) -> opaque),
        ("$Tensor$le", tensor_le, (i64, i64) -> opaque),
        ("$Tensor$gt", tensor_gt, (i64, i64) -> opaque),
        ("$Tensor$ge", tensor_ge, (i64, i64) -> opaque),

        // Axis-wise reductions
        ("$Tensor$sum_axis", tensor_sum_axis, (i64, u32) -> opaque),
        ("$Tensor$mean_axis", tensor_mean_axis, (i64, u32) -> opaque),
        ("$Tensor$max_axis", tensor_max_axis, (i64, u32) -> opaque),
        ("$Tensor$min_axis", tensor_min_axis, (i64, u32) -> opaque),

        // Shape operations
        ("$Tensor$flatten", tensor_flatten, (i64) -> opaque),
        ("$Tensor$transpose_axes", tensor_transpose_axes, (i64, u32, u32) -> opaque),

        // Conversion and utility
        ("$Tensor$to_list", tensor_to_list, (i64) -> i64),
        ("$Tensor$item", tensor_item, (i64) -> f64),

        // Higher-level operations
        ("$Tensor$softmax", tensor_softmax, (i64, i64) -> opaque),
        ("$Tensor$cross_entropy", tensor_cross_entropy, (i64, i64) -> f64),
        ("$Tensor$concat", tensor_concat, (i64, i64) -> opaque),
        ("$Tensor$stack", tensor_stack, (i64, i64) -> opaque),
        ("$Tensor$split", tensor_split, (i64, i64, i64) -> i64),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let shape = [2, 3];
        let tensor = tensor_new(shape.as_ptr(), 2, DType::F32 as u8);
        assert!(!tensor.is_null());

        unsafe {
            assert_eq!(tensor_ndim(tensor), 2);
            assert_eq!(tensor_shape(tensor, 0), 2);
            assert_eq!(tensor_shape(tensor, 1), 3);
            assert_eq!(tensor_numel(tensor), 6);
            assert!(tensor_is_contiguous(tensor));
        }

        tensor_free(tensor);
    }

    #[test]
    fn test_tensor_ones() {
        let shape = [3];
        let tensor = tensor_ones(shape.as_ptr(), 1, DType::F32 as u8);
        assert!(!tensor.is_null());

        for i in 0..3 {
            assert_eq!(tensor_get_f32(tensor, i), 1.0);
        }

        tensor_free(tensor);
    }

    #[test]
    fn test_tensor_arange() {
        let tensor = tensor_arange_f32(0.0, 5.0, 1.0);
        assert!(!tensor.is_null());

        assert_eq!(tensor_numel(tensor), 5);
        for i in 0..5 {
            assert_eq!(tensor_get_f32(tensor, i), i as f32);
        }

        tensor_free(tensor);
    }

    #[test]
    fn test_tensor_reshape() {
        let shape = [2, 3];
        let tensor = tensor_ones(shape.as_ptr(), 2, DType::F32 as u8);

        let new_shape = [6];
        let reshaped = tensor_reshape(tensor, new_shape.as_ptr(), 1);
        assert!(!reshaped.is_null());

        unsafe {
            assert_eq!(tensor_ndim(reshaped), 1);
            assert_eq!(tensor_shape(reshaped, 0), 6);
        }

        tensor_free(reshaped);
        tensor_free(tensor);
    }

    #[test]
    fn test_tensor_transpose() {
        let shape = [2, 3];
        let tensor = tensor_arange_f32(0.0, 6.0, 1.0);
        let reshaped = tensor_reshape(tensor, shape.as_ptr(), 2);
        let transposed = tensor_transpose(reshaped, 0, 1);

        unsafe {
            assert_eq!(tensor_shape(transposed, 0), 3);
            assert_eq!(tensor_shape(transposed, 1), 2);
        }

        tensor_free(transposed);
        tensor_free(reshaped);
        tensor_free(tensor);
    }

    #[test]
    fn test_tensor_sum() {
        let tensor = tensor_arange_f32(1.0, 6.0, 1.0);
        let sum = tensor_sum_f32(tensor);
        assert_eq!(sum, 15.0); // 1+2+3+4+5

        tensor_free(tensor);
    }

    #[test]
    fn test_tensor_slice() {
        let tensor = tensor_arange_f32(0.0, 10.0, 1.0);
        let sliced = tensor_slice(tensor, 0, 2, 7);

        assert_eq!(tensor_numel(sliced), 5);
        assert_eq!(tensor_get_f32(sliced, 0), 2.0);
        assert_eq!(tensor_get_f32(sliced, 4), 6.0);

        tensor_free(sliced);
        tensor_free(tensor);
    }

    #[test]
    fn test_tensor_clone() {
        let tensor = tensor_arange_f32(0.0, 5.0, 1.0);
        let cloned = tensor_clone(tensor);

        // Modify original
        tensor_set_f32(tensor, 0, 100.0);

        // Clone should be unchanged
        assert_eq!(tensor_get_f32(cloned, 0), 0.0);

        tensor_free(cloned);
        tensor_free(tensor);
    }
}
