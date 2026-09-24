//! ZRTL Environment Plugin
//!
//! Provides environment variables, command-line arguments, and process operations.
//!
//! ## Exported Symbols
//!
//! ### Environment Variables
//! - `$Env$get` - Get environment variable value
//! - `$Env$set` - Set environment variable
//! - `$Env$remove` - Remove environment variable
//! - `$Env$has` - Check if environment variable exists
//!
//! ### Command-line Arguments
//! - `$Env$args_count` - Get number of command-line arguments
//! - `$Env$arg` - Get argument at index
//! - `$Env$exe_path` - Get path to current executable
//!
//! ### Process
//! - `$Env$exit` - Exit the process with status code
//! - `$Env$pid` - Get current process ID
//! - `$Env$home_dir` - Get user's home directory
//! - `$Env$temp_dir` - Get system temp directory

use std::env;
use zrtl::{
    zrtl_plugin,
    StringConstPtr, StringPtr,
    string_new,
};

// ============================================================================
// Helper
// ============================================================================

/// Convert a ZRTL string pointer to a Rust string slice
unsafe fn zrtl_string_to_str<'a>(s: StringConstPtr) -> Option<&'a str> {
    zrtl::string_as_str(s)
}

// ============================================================================
// Environment Variable Functions
// ============================================================================

/// Get an environment variable value
///
/// Returns a ZRTL string pointer, or null if not found.
/// Caller must free with `string_free`.
///
/// # Safety
/// The key must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn env_get(key: StringConstPtr) -> StringPtr {
    let key_str = match zrtl_string_to_str(key) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };

    match env::var(key_str) {
        Ok(value) => string_new(&value),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Set an environment variable
///
/// Returns 0 on success.
///
/// # Safety
/// Both key and value must be valid ZRTL string pointers.
#[no_mangle]
pub unsafe extern "C" fn env_set(key: StringConstPtr, value: StringConstPtr) -> i32 {
    let key_str = match zrtl_string_to_str(key) {
        Some(s) => s,
        None => return -1,
    };
    let value_str = match zrtl_string_to_str(value) {
        Some(s) => s,
        None => return -1,
    };

    env::set_var(key_str, value_str);
    0
}

/// Remove an environment variable
///
/// # Safety
/// The key must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn env_remove(key: StringConstPtr) {
    if let Some(key_str) = zrtl_string_to_str(key) {
        env::remove_var(key_str);
    }
}

/// Check if an environment variable exists
///
/// Returns 1 if exists, 0 if not.
///
/// # Safety
/// The key must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn env_has(key: StringConstPtr) -> i32 {
    match zrtl_string_to_str(key) {
        Some(key_str) => if env::var(key_str).is_ok() { 1 } else { 0 },
        None => 0,
    }
}

// ============================================================================
// Command-line Argument Functions
// ============================================================================

/// Get the number of command-line arguments
#[no_mangle]
pub extern "C" fn env_args_count() -> i32 {
    env::args().count() as i32
}

/// Get command-line argument at index
///
/// Returns a ZRTL string pointer, or null if index out of bounds.
/// Caller must free with `string_free`.
#[no_mangle]
pub extern "C" fn env_arg(index: i32) -> StringPtr {
    if index < 0 {
        return std::ptr::null_mut();
    }

    match env::args().nth(index as usize) {
        Some(arg) => string_new(&arg),
        None => std::ptr::null_mut(),
    }
}

/// Get path to current executable
///
/// Returns a ZRTL string pointer, or null on error.
/// Caller must free with `string_free`.
#[no_mangle]
pub extern "C" fn env_exe_path() -> StringPtr {
    match env::current_exe() {
        Ok(path) => {
            match path.to_str() {
                Some(s) => string_new(s),
                None => std::ptr::null_mut(),
            }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

// ============================================================================
// Process Functions
// ============================================================================

/// Exit the process with a status code
///
/// This function does not return.
#[no_mangle]
pub extern "C" fn env_exit(code: i32) -> ! {
    std::process::exit(code)
}

/// Get current process ID
#[no_mangle]
pub extern "C" fn env_pid() -> u32 {
    std::process::id()
}

/// Get user's home directory
///
/// Returns a ZRTL string pointer, or null if not found.
/// Caller must free with `string_free`.
#[no_mangle]
pub extern "C" fn env_home_dir() -> StringPtr {
    // Try HOME on Unix, USERPROFILE on Windows
    #[cfg(unix)]
    if let Ok(home) = env::var("HOME") {
        return string_new(&home);
    }

    #[cfg(windows)]
    if let Ok(home) = env::var("USERPROFILE") {
        return string_new(&home);
    }

    std::ptr::null_mut()
}

/// Get system temp directory
///
/// Returns a ZRTL string pointer.
/// Caller must free with `string_free`.
#[no_mangle]
pub extern "C" fn env_temp_dir() -> StringPtr {
    let temp = env::temp_dir();
    match temp.to_str() {
        Some(s) => string_new(s),
        None => std::ptr::null_mut(),
    }
}

/// Get current working directory
///
/// Returns a ZRTL string pointer, or null on error.
/// Caller must free with `string_free`.
#[no_mangle]
pub extern "C" fn env_current_dir() -> StringPtr {
    match env::current_dir() {
        Ok(path) => {
            match path.to_str() {
                Some(s) => string_new(s),
                None => std::ptr::null_mut(),
            }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Get username of current user
///
/// Returns a ZRTL string pointer, or null if not found.
/// Caller must free with `string_free`.
#[no_mangle]
pub extern "C" fn env_username() -> StringPtr {
    // Try USER on Unix, USERNAME on Windows
    #[cfg(unix)]
    if let Ok(user) = env::var("USER") {
        return string_new(&user);
    }

    #[cfg(windows)]
    if let Ok(user) = env::var("USERNAME") {
        return string_new(&user);
    }

    std::ptr::null_mut()
}

/// Get operating system name
///
/// Returns "macos", "linux", "windows", or "unknown"
#[no_mangle]
pub extern "C" fn env_os() -> StringPtr {
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "freebsd") {
        "freebsd"
    } else if cfg!(target_os = "android") {
        "android"
    } else if cfg!(target_os = "ios") {
        "ios"
    } else {
        "unknown"
    };
    string_new(os)
}

/// Get CPU architecture
///
/// Returns "x86_64", "aarch64", "x86", "arm", or "unknown"
#[no_mangle]
pub extern "C" fn env_arch() -> StringPtr {
    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "x86") {
        "x86"
    } else if cfg!(target_arch = "arm") {
        "arm"
    } else if cfg!(target_arch = "wasm32") {
        "wasm32"
    } else {
        "unknown"
    };
    string_new(arch)
}

// ============================================================================
// Plugin Export
// ============================================================================

zrtl_plugin! {
    name: "zrtl_env",
    symbols: [
        // Environment variables
        ("$Env$get", env_get),
        ("$Env$set", env_set),
        ("$Env$remove", env_remove),
        ("$Env$has", env_has),

        // Command-line arguments
        ("$Env$args_count", env_args_count),
        ("$Env$arg", env_arg),
        ("$Env$exe_path", env_exe_path),

        // Process
        ("$Env$exit", env_exit),
        ("$Env$pid", env_pid),

        // Directories
        ("$Env$home_dir", env_home_dir),
        ("$Env$temp_dir", env_temp_dir),
        ("$Env$current_dir", env_current_dir),

        // System info
        ("$Env$username", env_username),
        ("$Env$os", env_os),
        ("$Env$arch", env_arch),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use zrtl::{string_as_str, string_free, string_new};

    #[test]
    fn test_env_set_get() {
        let key = string_new("ZRTL_TEST_VAR");
        let value = string_new("test_value_123");

        unsafe {
            // Set
            env_set(key, value);

            // Get
            let result = env_get(key);
            assert!(!result.is_null());
            assert_eq!(string_as_str(result), Some("test_value_123"));

            // Has
            assert_eq!(env_has(key), 1);

            // Remove
            env_remove(key);
            assert_eq!(env_has(key), 0);

            string_free(key);
            string_free(value);
            string_free(result);
        }
    }

    #[test]
    fn test_env_args() {
        let count = env_args_count();
        assert!(count >= 1); // At least the executable path

        let arg0 = env_arg(0);
        assert!(!arg0.is_null());
        unsafe {
            string_free(arg0);
        }
    }

    #[test]
    fn test_pid() {
        let pid = env_pid();
        assert!(pid > 0);
    }

    #[test]
    fn test_temp_dir() {
        let temp = env_temp_dir();
        assert!(!temp.is_null());
        unsafe {
            let s = string_as_str(temp);
            assert!(s.is_some());
            assert!(!s.unwrap().is_empty());
            string_free(temp);
        }
    }

    #[test]
    fn test_os_and_arch() {
        let os = env_os();
        let arch = env_arch();

        assert!(!os.is_null());
        assert!(!arch.is_null());

        unsafe {
            let os_str = string_as_str(os).unwrap();
            let arch_str = string_as_str(arch).unwrap();

            // Should be one of the known values
            assert!(["macos", "linux", "windows", "freebsd", "android", "ios", "unknown"].contains(&os_str));
            assert!(["x86_64", "aarch64", "x86", "arm", "wasm32", "unknown"].contains(&arch_str));

            string_free(os);
            string_free(arch);
        }
    }
}
