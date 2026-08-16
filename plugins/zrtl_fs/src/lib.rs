//! ZRTL File System Plugin
//!
//! Provides file system operations for Zyntax-based languages using
//! the ZRTL SDK types (StringPtr, ArrayPtr, etc.).
//!
//! ## Exported Symbols
//!
//! ### File Operations
//! - `$FS$read_file` - Read entire file as ZRTL string
//! - `$FS$read_bytes` - Read file as byte array
//! - `$FS$write_file` - Write ZRTL string to file
//! - `$FS$write_bytes` - Write byte array to file
//! - `$FS$append_file` - Append ZRTL string to file
//!
//! ### File Info
//! - `$FS$exists` - Check if path exists
//! - `$FS$is_file` - Check if path is a file
//! - `$FS$is_dir` - Check if path is a directory
//! - `$FS$file_size` - Get file size in bytes
//!
//! ### Directory Operations
//! - `$FS$create_dir` - Create a directory
//! - `$FS$create_dir_all` - Create directory and parents
//! - `$FS$remove_file` - Delete a file
//! - `$FS$remove_dir` - Delete an empty directory
//! - `$FS$remove_dir_all` - Delete directory recursively
//!
//! ### Path Operations
//! - `$FS$current_dir` - Get current working directory
//! - `$FS$set_current_dir` - Change current working directory
//! - `$FS$canonicalize` - Get absolute path

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use zrtl::{
    array_data, array_length, array_new, array_push, string_data, string_length, string_new,
    zrtl_plugin, ArrayPtr, StringConstPtr, StringPtr,
};

// ============================================================================
// Helper: Convert ZRTL string to Rust Path
// ============================================================================

/// Convert a ZRTL string pointer to a Rust string slice
///
/// # Safety
/// The pointer must be a valid ZRTL string pointer.
unsafe fn zrtl_string_to_str<'a>(s: StringConstPtr) -> Option<&'a str> {
    if s.is_null() {
        return None;
    }
    let len = string_length(s) as usize;
    let data = string_data(s);
    if len == 0 || data.is_null() {
        return Some("");
    }
    let bytes = std::slice::from_raw_parts(data, len);
    std::str::from_utf8(bytes).ok()
}

// ============================================================================
// File Read Operations
// ============================================================================

/// Read entire file contents as a ZRTL string
///
/// Returns a ZRTL string pointer, or null on error.
/// Caller must free with `string_free`.
///
/// # Safety
/// The path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_read_file(path: StringConstPtr) -> StringPtr {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };

    match fs::read_to_string(path_str) {
        Ok(contents) => string_new(&contents),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Read file as raw bytes into a ZRTL array of u8
///
/// Returns an ArrayPtr to `[i32 cap][i32 len][u8...]`, or null on error.
/// Caller must free with `array_free`.
///
/// # Safety
/// The path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_read_bytes(path: StringConstPtr) -> ArrayPtr {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };

    match fs::read(path_str) {
        Ok(bytes) => {
            // Create array with capacity for all bytes
            let arr = array_new::<u8>(bytes.len());
            if arr.is_null() {
                return std::ptr::null_mut();
            }
            // Push each byte
            for byte in bytes {
                array_push(arr, byte);
            }
            arr
        }
        Err(_) => std::ptr::null_mut(),
    }
}

// ============================================================================
// File Write Operations
// ============================================================================

/// Write a ZRTL string to a file (creates or overwrites)
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// Both pointers must be valid ZRTL string pointers.
#[no_mangle]
pub unsafe extern "C" fn fs_write_file(path: StringConstPtr, contents: StringConstPtr) -> i32 {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return -1,
    };

    let len = string_length(contents) as usize;
    let data = string_data(contents);

    if data.is_null() && len > 0 {
        return -1;
    }

    let bytes = if len > 0 {
        std::slice::from_raw_parts(data, len)
    } else {
        &[]
    };

    match fs::write(path_str, bytes) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Write a byte array to a file (creates or overwrites)
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// Path must be a valid ZRTL string, bytes must be a valid ZRTL array.
#[no_mangle]
pub unsafe extern "C" fn fs_write_bytes(path: StringConstPtr, bytes: ArrayPtr) -> i32 {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return -1,
    };

    if bytes.is_null() {
        return -1;
    }

    let len = array_length(bytes) as usize;
    let data = array_data::<u8>(bytes);

    let slice = if len > 0 && !data.is_null() {
        std::slice::from_raw_parts(data, len)
    } else {
        &[]
    };

    match fs::write(path_str, slice) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Append a ZRTL string to a file
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// Both pointers must be valid ZRTL string pointers.
#[no_mangle]
pub unsafe extern "C" fn fs_append_file(path: StringConstPtr, contents: StringConstPtr) -> i32 {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return -1,
    };

    let len = string_length(contents) as usize;
    let data = string_data(contents);

    let bytes = if len > 0 && !data.is_null() {
        std::slice::from_raw_parts(data, len)
    } else {
        &[]
    };

    match OpenOptions::new().append(true).create(true).open(path_str) {
        Ok(mut file) => match file.write_all(bytes) {
            Ok(()) => 0,
            Err(_) => -1,
        },
        Err(_) => -1,
    }
}

// ============================================================================
// File Info Operations
// ============================================================================

/// Check if a path exists
///
/// Returns 1 if exists, 0 if not.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_exists(path: StringConstPtr) -> i32 {
    match zrtl_string_to_str(path) {
        Some(s) => {
            if Path::new(s).exists() {
                1
            } else {
                0
            }
        }
        None => 0,
    }
}

/// Check if path is a file
///
/// Returns 1 if file, 0 if not.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_is_file(path: StringConstPtr) -> i32 {
    match zrtl_string_to_str(path) {
        Some(s) => {
            if Path::new(s).is_file() {
                1
            } else {
                0
            }
        }
        None => 0,
    }
}

/// Check if path is a directory
///
/// Returns 1 if directory, 0 if not.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_is_dir(path: StringConstPtr) -> i32 {
    match zrtl_string_to_str(path) {
        Some(s) => {
            if Path::new(s).is_dir() {
                1
            } else {
                0
            }
        }
        None => 0,
    }
}

/// Get file size in bytes
///
/// Returns file size, or -1 on error.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_file_size(path: StringConstPtr) -> i64 {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return -1,
    };

    match fs::metadata(path_str) {
        Ok(meta) => meta.len() as i64,
        Err(_) => -1,
    }
}

/// Check if path is a symbolic link
///
/// Returns 1 if symlink, 0 if not.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_is_symlink(path: StringConstPtr) -> i32 {
    match zrtl_string_to_str(path) {
        Some(s) => {
            if Path::new(s).is_symlink() {
                1
            } else {
                0
            }
        }
        None => 0,
    }
}

// ============================================================================
// Directory Operations
// ============================================================================

/// Create a directory
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_create_dir(path: StringConstPtr) -> i32 {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return -1,
    };

    match fs::create_dir(path_str) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Create a directory and all parent directories
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_create_dir_all(path: StringConstPtr) -> i32 {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return -1,
    };

    match fs::create_dir_all(path_str) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Remove a file
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_remove_file(path: StringConstPtr) -> i32 {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return -1,
    };

    match fs::remove_file(path_str) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Remove an empty directory
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_remove_dir(path: StringConstPtr) -> i32 {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return -1,
    };

    match fs::remove_dir(path_str) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Remove a directory and all its contents recursively
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_remove_dir_all(path: StringConstPtr) -> i32 {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return -1,
    };

    match fs::remove_dir_all(path_str) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Rename/move a file or directory
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// Both paths must be valid ZRTL string pointers.
#[no_mangle]
pub unsafe extern "C" fn fs_rename(from: StringConstPtr, to: StringConstPtr) -> i32 {
    let from_str = match zrtl_string_to_str(from) {
        Some(s) => s,
        None => return -1,
    };
    let to_str = match zrtl_string_to_str(to) {
        Some(s) => s,
        None => return -1,
    };

    match fs::rename(from_str, to_str) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Copy a file
///
/// Returns bytes copied on success, -1 on error.
///
/// # Safety
/// Both paths must be valid ZRTL string pointers.
#[no_mangle]
pub unsafe extern "C" fn fs_copy(from: StringConstPtr, to: StringConstPtr) -> i64 {
    let from_str = match zrtl_string_to_str(from) {
        Some(s) => s,
        None => return -1,
    };
    let to_str = match zrtl_string_to_str(to) {
        Some(s) => s,
        None => return -1,
    };

    match fs::copy(from_str, to_str) {
        Ok(bytes) => bytes as i64,
        Err(_) => -1,
    }
}

// ============================================================================
// Path Operations
// ============================================================================

/// Get the current working directory
///
/// Returns a ZRTL string pointer, or null on error.
/// Caller must free with `string_free`.
#[no_mangle]
pub extern "C" fn fs_current_dir() -> StringPtr {
    match std::env::current_dir() {
        Ok(path) => match path.to_str() {
            Some(s) => string_new(s),
            None => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Set the current working directory
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_set_current_dir(path: StringConstPtr) -> i32 {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return -1,
    };

    match std::env::set_current_dir(path_str) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Get the absolute/canonical path
///
/// Returns a ZRTL string pointer, or null on error.
/// Caller must free with `string_free`.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_canonicalize(path: StringConstPtr) -> StringPtr {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };

    match fs::canonicalize(path_str) {
        Ok(abs_path) => match abs_path.to_str() {
            Some(s) => string_new(s),
            None => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Get the file extension
///
/// Returns a ZRTL string pointer (empty if no extension), or null on error.
/// Caller must free with `string_free`.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_extension(path: StringConstPtr) -> StringPtr {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };

    let p = Path::new(path_str);
    match p.extension().and_then(|e| e.to_str()) {
        Some(ext) => string_new(ext),
        None => string_new(""),
    }
}

/// Get the file name (last component of path)
///
/// Returns a ZRTL string pointer, or null on error.
/// Caller must free with `string_free`.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_file_name(path: StringConstPtr) -> StringPtr {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };

    let p = Path::new(path_str);
    match p.file_name().and_then(|n| n.to_str()) {
        Some(name) => string_new(name),
        None => std::ptr::null_mut(),
    }
}

/// Get the parent directory
///
/// Returns a ZRTL string pointer, or null if no parent.
/// Caller must free with `string_free`.
///
/// # Safety
/// Path must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn fs_parent(path: StringConstPtr) -> StringPtr {
    let path_str = match zrtl_string_to_str(path) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };

    let p = Path::new(path_str);
    match p.parent().and_then(|p| p.to_str()) {
        Some(parent) => string_new(parent),
        None => std::ptr::null_mut(),
    }
}

// ============================================================================
// Plugin Export
// ============================================================================

zrtl_plugin! {
    name: "zrtl_fs",
    symbols: [
        // File read
        ("$FS$read_file", fs_read_file),
        ("$FS$read_bytes", fs_read_bytes),

        // File write
        ("$FS$write_file", fs_write_file),
        ("$FS$write_bytes", fs_write_bytes),
        ("$FS$append_file", fs_append_file),

        // File info
        ("$FS$exists", fs_exists),
        ("$FS$is_file", fs_is_file),
        ("$FS$is_dir", fs_is_dir),
        ("$FS$is_symlink", fs_is_symlink),
        ("$FS$file_size", fs_file_size),

        // Directory operations
        ("$FS$create_dir", fs_create_dir),
        ("$FS$create_dir_all", fs_create_dir_all),
        ("$FS$remove_file", fs_remove_file),
        ("$FS$remove_dir", fs_remove_dir),
        ("$FS$remove_dir_all", fs_remove_dir_all),
        ("$FS$rename", fs_rename),
        ("$FS$copy", fs_copy),

        // Path operations
        ("$FS$current_dir", fs_current_dir),
        ("$FS$set_current_dir", fs_set_current_dir),
        ("$FS$canonicalize", fs_canonicalize),
        ("$FS$extension", fs_extension),
        ("$FS$file_name", fs_file_name),
        ("$FS$parent", fs_parent),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use zrtl::{string_as_str, string_free, string_new};

    #[test]
    fn test_current_dir() {
        let cwd = fs_current_dir();
        assert!(!cwd.is_null());
        unsafe {
            let s = string_as_str(cwd);
            assert!(s.is_some());
            assert!(!s.unwrap().is_empty());
            string_free(cwd);
        }
    }

    #[test]
    fn test_exists() {
        let path = string_new(".");
        unsafe {
            assert_eq!(fs_exists(path), 1);
            assert_eq!(fs_is_dir(path), 1);
            assert_eq!(fs_is_file(path), 0);
            string_free(path);
        }
    }

    #[test]
    fn test_read_write_file() {
        use std::env::temp_dir;

        let temp = temp_dir().join("zrtl_test_file.txt");
        let temp_str = temp.to_str().unwrap();
        let path = string_new(temp_str);
        let contents = string_new("Hello, ZRTL!");

        unsafe {
            // Write
            let result = fs_write_file(path, contents);
            assert_eq!(result, 0);

            // Verify exists
            assert_eq!(fs_exists(path), 1);
            assert_eq!(fs_is_file(path), 1);

            // Read back
            let read_contents = fs_read_file(path);
            assert!(!read_contents.is_null());
            assert_eq!(string_as_str(read_contents), Some("Hello, ZRTL!"));

            // Cleanup
            fs_remove_file(path);
            assert_eq!(fs_exists(path), 0);

            string_free(path);
            string_free(contents);
            string_free(read_contents);
        }
    }

    #[test]
    fn test_path_operations() {
        let path = string_new("/foo/bar/file.txt");
        unsafe {
            let ext = fs_extension(path);
            let name = fs_file_name(path);
            let parent = fs_parent(path);

            assert_eq!(string_as_str(ext), Some("txt"));
            assert_eq!(string_as_str(name), Some("file.txt"));
            assert_eq!(string_as_str(parent), Some("/foo/bar"));

            string_free(ext);
            string_free(name);
            string_free(parent);
            string_free(path);
        }
    }
}
