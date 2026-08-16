//! # ZRTL Model Plugin
//!
//! Provides model loading and serialization for machine learning workloads.
//! Supports SafeTensors format (HuggingFace standard) for loading pre-trained models.
//!
//! ## Features
//!
//! - Load SafeTensors model files
//! - Memory-mapped access for large models
//! - Tensor inspection and extraction
//! - Model metadata access
//!
//! ## Example
//!
//! ```text
//! let model = model_load("model.safetensors");
//! let num_tensors = model_num_tensors(model);
//! let weights = model_get_tensor(model, "model.layers.0.weight");
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use memmap2::Mmap;
use safetensors::SafeTensors;
use zrtl::zrtl_plugin;

// ============================================================================
// Model Handle
// ============================================================================

/// Model format enumeration
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    SafeTensors = 0,
    Unknown = 255,
}

/// Data type in model files
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelDType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
    I8 = 4,
    I16 = 5,
    I32 = 6,
    I64 = 7,
    U8 = 8,
    U16 = 9,
    U32 = 10,
    U64 = 11,
    Bool = 12,
    Unknown = 255,
}

impl From<safetensors::Dtype> for ModelDType {
    fn from(dtype: safetensors::Dtype) -> Self {
        match dtype {
            safetensors::Dtype::F32 => ModelDType::F32,
            safetensors::Dtype::F64 => ModelDType::F64,
            safetensors::Dtype::F16 => ModelDType::F16,
            safetensors::Dtype::BF16 => ModelDType::BF16,
            safetensors::Dtype::I8 => ModelDType::I8,
            safetensors::Dtype::I16 => ModelDType::I16,
            safetensors::Dtype::I32 => ModelDType::I32,
            safetensors::Dtype::I64 => ModelDType::I64,
            safetensors::Dtype::U8 => ModelDType::U8,
            safetensors::Dtype::U16 => ModelDType::U16,
            safetensors::Dtype::U32 => ModelDType::U32,
            safetensors::Dtype::U64 => ModelDType::U64,
            safetensors::Dtype::BOOL => ModelDType::Bool,
            _ => ModelDType::Unknown,
        }
    }
}

/// Tensor info structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (pointer to internal storage)
    pub name_ptr: *const u8,
    pub name_len: usize,
    /// Shape dimensions
    pub shape: [usize; 8],
    pub ndim: u32,
    /// Data type
    pub dtype: ModelDType,
    /// Size in bytes
    pub size_bytes: usize,
}

/// Internal model storage
struct ModelData {
    /// Memory-mapped file
    mmap: Mmap,
    /// Format
    format: ModelFormat,
    /// Tensor names for lookup
    tensor_names: Vec<String>,
    /// Metadata
    metadata: HashMap<String, String>,
}

impl ModelData {
    fn from_safetensors(mmap: Mmap) -> Option<Self> {
        // Validate the safetensors format
        let tensors = match SafeTensors::deserialize(&mmap) {
            Ok(t) => t,
            Err(_) => return None,
        };

        let tensor_names: Vec<String> = tensors.names().iter().map(|s| s.to_string()).collect();

        // Metadata is not directly accessible in safetensors 0.4, use empty map
        let metadata = HashMap::new();

        Some(ModelData {
            mmap,
            format: ModelFormat::SafeTensors,
            tensor_names,
            metadata,
        })
    }

    fn num_tensors(&self) -> usize {
        self.tensor_names.len()
    }

    fn tensor_name(&self, index: usize) -> Option<&str> {
        self.tensor_names.get(index).map(|s| s.as_str())
    }

    fn get_tensor_info(&self, name: &str) -> Option<TensorInfo> {
        let tensors = SafeTensors::deserialize(&self.mmap).ok()?;
        let view = tensors.tensor(name).ok()?;

        let mut shape = [0usize; 8];
        let ndim = view.shape().len().min(8);
        for (i, &dim) in view.shape().iter().take(8).enumerate() {
            shape[i] = dim;
        }

        Some(TensorInfo {
            name_ptr: std::ptr::null(), // Will be set by caller
            name_len: 0,
            shape,
            ndim: ndim as u32,
            dtype: view.dtype().into(),
            size_bytes: view.data().len(),
        })
    }

    fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let tensors = SafeTensors::deserialize(&self.mmap).ok()?;
        let view = tensors.tensor(name).ok()?;
        Some(view.data())
    }

    fn total_params(&self) -> u64 {
        let tensors = match SafeTensors::deserialize(&self.mmap) {
            Ok(t) => t,
            Err(_) => return 0,
        };

        let mut total = 0u64;
        for name in tensors.names() {
            if let Ok(view) = tensors.tensor(name) {
                let numel: usize = view.shape().iter().product();
                total += numel as u64;
            }
        }
        total
    }

    fn total_bytes(&self) -> u64 {
        let tensors = match SafeTensors::deserialize(&self.mmap) {
            Ok(t) => t,
            Err(_) => return 0,
        };

        let mut total = 0u64;
        for name in tensors.names() {
            if let Ok(view) = tensors.tensor(name) {
                total += view.data().len() as u64;
            }
        }
        total
    }
}

/// Global model storage
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);
static MODEL_STORE: Mutex<Option<HashMap<u64, ModelData>>> = Mutex::new(None);

fn init_store() {
    let mut store = MODEL_STORE.lock().unwrap();
    if store.is_none() {
        *store = Some(HashMap::new());
    }
}

fn insert_model(model: ModelData) -> u64 {
    init_store();
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    let mut store = MODEL_STORE.lock().unwrap();
    if let Some(ref mut map) = *store {
        map.insert(handle, model);
    }
    handle
}

fn get_model<F, R>(handle: u64, f: F) -> R
where
    F: FnOnce(Option<&ModelData>) -> R,
{
    init_store();
    let store = MODEL_STORE.lock().unwrap();
    if let Some(ref map) = *store {
        f(map.get(&handle))
    } else {
        f(None)
    }
}

fn take_model(handle: u64) -> Option<ModelData> {
    init_store();
    let mut store = MODEL_STORE.lock().unwrap();
    if let Some(ref mut map) = *store {
        map.remove(&handle)
    } else {
        None
    }
}

pub type ModelHandle = u64;
pub const MODEL_NULL: ModelHandle = 0;

// ============================================================================
// Loading
// ============================================================================

/// Load model from file path
#[no_mangle]
pub extern "C" fn model_load(path: *const u8, path_len: usize) -> ModelHandle {
    if path.is_null() || path_len == 0 {
        return MODEL_NULL;
    }

    let path_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(path, path_len)) {
            Ok(s) => s,
            Err(_) => return MODEL_NULL,
        }
    };

    // Open and memory-map the file
    let file = match File::open(path_str) {
        Ok(f) => f,
        Err(_) => return MODEL_NULL,
    };

    let mmap = unsafe {
        match Mmap::map(&file) {
            Ok(m) => m,
            Err(_) => return MODEL_NULL,
        }
    };

    // Detect format and parse
    let model = if path_str.ends_with(".safetensors") {
        ModelData::from_safetensors(mmap)
    } else {
        // Try SafeTensors by default
        ModelData::from_safetensors(mmap)
    };

    match model {
        Some(m) => insert_model(m),
        None => MODEL_NULL,
    }
}

/// Free model handle
#[no_mangle]
pub extern "C" fn model_free(handle: ModelHandle) {
    take_model(handle);
}

// ============================================================================
// Model Info
// ============================================================================

/// Get model format
#[no_mangle]
pub extern "C" fn model_format(handle: ModelHandle) -> u8 {
    get_model(handle, |m| {
        m.map(|m| m.format as u8)
            .unwrap_or(ModelFormat::Unknown as u8)
    })
}

/// Get number of tensors in model
#[no_mangle]
pub extern "C" fn model_num_tensors(handle: ModelHandle) -> usize {
    get_model(handle, |m| m.map(|m| m.num_tensors()).unwrap_or(0))
}

/// Get total number of parameters
#[no_mangle]
pub extern "C" fn model_total_params(handle: ModelHandle) -> u64 {
    get_model(handle, |m| m.map(|m| m.total_params()).unwrap_or(0))
}

/// Get total size in bytes
#[no_mangle]
pub extern "C" fn model_total_bytes(handle: ModelHandle) -> u64 {
    get_model(handle, |m| m.map(|m| m.total_bytes()).unwrap_or(0))
}

/// Get tensor name by index
/// Returns length of name written to output buffer
#[no_mangle]
pub extern "C" fn model_tensor_name(
    handle: ModelHandle,
    index: usize,
    output: *mut u8,
    output_capacity: usize,
) -> usize {
    if output.is_null() || output_capacity == 0 {
        return 0;
    }

    get_model(handle, |m| {
        if let Some(model) = m {
            if let Some(name) = model.tensor_name(index) {
                let bytes = name.as_bytes();
                let n = bytes.len().min(output_capacity);
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), output, n);
                }
                return n;
            }
        }
        0
    })
}

/// Get tensor info by name
#[no_mangle]
pub extern "C" fn model_tensor_info(
    handle: ModelHandle,
    name: *const u8,
    name_len: usize,
    info: *mut TensorInfo,
) -> bool {
    if name.is_null() || name_len == 0 || info.is_null() {
        return false;
    }

    let name_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(name, name_len)) {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    get_model(handle, |m| {
        if let Some(model) = m {
            if let Some(tensor_info) = model.get_tensor_info(name_str) {
                unsafe {
                    *info = tensor_info;
                }
                return true;
            }
        }
        false
    })
}

/// Get tensor data pointer and length
/// Returns false if tensor not found
#[no_mangle]
pub extern "C" fn model_tensor_data(
    handle: ModelHandle,
    name: *const u8,
    name_len: usize,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
) -> bool {
    if name.is_null() || name_len == 0 || out_ptr.is_null() || out_len.is_null() {
        return false;
    }

    let name_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(name, name_len)) {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    get_model(handle, |m| {
        if let Some(model) = m {
            if let Some(data) = model.get_tensor_data(name_str) {
                unsafe {
                    *out_ptr = data.as_ptr();
                    *out_len = data.len();
                }
                return true;
            }
        }
        false
    })
}

/// Check if model contains a tensor with given name
#[no_mangle]
pub extern "C" fn model_has_tensor(handle: ModelHandle, name: *const u8, name_len: usize) -> bool {
    if name.is_null() || name_len == 0 {
        return false;
    }

    let name_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(name, name_len)) {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    get_model(handle, |m| {
        m.map(|m| m.tensor_names.iter().any(|n| n == name_str))
            .unwrap_or(false)
    })
}

// ============================================================================
// Metadata
// ============================================================================

/// Get metadata value by key
#[no_mangle]
pub extern "C" fn model_metadata_get(
    handle: ModelHandle,
    key: *const u8,
    key_len: usize,
    output: *mut u8,
    output_capacity: usize,
) -> usize {
    if key.is_null() || key_len == 0 || output.is_null() || output_capacity == 0 {
        return 0;
    }

    let key_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(key, key_len)) {
            Ok(s) => s,
            Err(_) => return 0,
        }
    };

    get_model(handle, |m| {
        if let Some(model) = m {
            if let Some(value) = model.metadata.get(key_str) {
                let bytes = value.as_bytes();
                let n = bytes.len().min(output_capacity);
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), output, n);
                }
                return n;
            }
        }
        0
    })
}

// ============================================================================
// Tensor Copying
// ============================================================================

/// Copy tensor data to f32 buffer (with type conversion if needed)
/// Returns number of elements copied
#[no_mangle]
pub extern "C" fn model_copy_tensor_f32(
    handle: ModelHandle,
    name: *const u8,
    name_len: usize,
    output: *mut f32,
    output_capacity: usize,
) -> usize {
    if name.is_null() || name_len == 0 || output.is_null() || output_capacity == 0 {
        return 0;
    }

    let name_str = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(name, name_len)) {
            Ok(s) => s,
            Err(_) => return 0,
        }
    };

    get_model(handle, |m| {
        let model = match m {
            Some(m) => m,
            None => return 0,
        };

        let tensors = match SafeTensors::deserialize(&model.mmap) {
            Ok(t) => t,
            Err(_) => return 0,
        };

        let view = match tensors.tensor(name_str) {
            Ok(v) => v,
            Err(_) => return 0,
        };

        let numel: usize = view.shape().iter().product();
        let n = numel.min(output_capacity);
        let data = view.data();

        match view.dtype() {
            safetensors::Dtype::F32 => {
                let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, numel) };
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), output, n);
                }
            }
            safetensors::Dtype::F16 => {
                // Simple F16 to F32 conversion
                for i in 0..n {
                    let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                    unsafe {
                        *output.add(i) = f16_to_f32(bits);
                    }
                }
            }
            safetensors::Dtype::BF16 => {
                // BF16 to F32 conversion
                for i in 0..n {
                    let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                    unsafe {
                        *output.add(i) = bf16_to_f32(bits);
                    }
                }
            }
            safetensors::Dtype::F64 => {
                let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, numel) };
                for i in 0..n {
                    unsafe {
                        *output.add(i) = src[i] as f32;
                    }
                }
            }
            _ => return 0,
        }

        n
    })
}

/// Convert F16 bits to F32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut e = -14i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = ((e + 127) as u32) & 0xFF;
            let f32_mant = m << 13;
            f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
        }
    } else if exp == 0x1F {
        // Inf or NaN
        let f32_mant = mant << 13;
        f32::from_bits((sign << 31) | (0xFF << 23) | f32_mant)
    } else {
        // Normal
        let f32_exp = (exp + 112) & 0xFF;
        let f32_mant = mant << 13;
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
    }
}

/// Convert BF16 bits to F32
fn bf16_to_f32(bits: u16) -> f32 {
    // BF16 is just the upper 16 bits of F32
    f32::from_bits((bits as u32) << 16)
}

// ============================================================================
// Plugin Registration
// ============================================================================

zrtl_plugin! {
    name: "model",
    symbols: [
        // Loading
        ("$Model$load", model_load),
        ("$Model$free", model_free),

        // Info
        ("$Model$format", model_format),
        ("$Model$num_tensors", model_num_tensors),
        ("$Model$total_params", model_total_params),
        ("$Model$total_bytes", model_total_bytes),

        // Tensor access
        ("$Model$tensor_name", model_tensor_name),
        ("$Model$tensor_info", model_tensor_info),
        ("$Model$tensor_data", model_tensor_data),
        ("$Model$has_tensor", model_has_tensor),
        ("$Model$copy_tensor_f32", model_copy_tensor_f32),

        // Metadata
        ("$Model$metadata_get", model_metadata_get),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        // Test zero
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);

        // Test one
        let one_f16 = 0x3C00; // 1.0 in F16
        assert!((f16_to_f32(one_f16) - 1.0).abs() < 0.001);

        // Test negative one
        let neg_one_f16 = 0xBC00; // -1.0 in F16
        assert!((f16_to_f32(neg_one_f16) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bf16_conversion() {
        // Test one
        let one_bf16 = 0x3F80; // Upper 16 bits of 1.0f32
        assert!((bf16_to_f32(one_bf16) - 1.0).abs() < 0.001);

        // Test two
        let two_bf16 = 0x4000; // Upper 16 bits of 2.0f32
        assert!((bf16_to_f32(two_bf16) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_model_handle_null() {
        // Operations on null handle should return defaults
        assert_eq!(model_format(MODEL_NULL), ModelFormat::Unknown as u8);
        assert_eq!(model_num_tensors(MODEL_NULL), 0);
        assert_eq!(model_total_params(MODEL_NULL), 0);
    }
}
