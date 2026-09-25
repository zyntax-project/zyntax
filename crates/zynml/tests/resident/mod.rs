//! The resident size of this process, as the operating system counts
//! it: the resident set on Linux and macOS, the working set on Windows.
//!
//! A test that bounds memory growth includes this with `mod resident;`.
//! A reading that cannot be taken panics rather than returning 0, since
//! a bound on growth measured from 0 to 0 passes whatever the program
//! did.

/// Resident bytes of this process.
pub fn bytes() -> u64 {
    match read() {
        Ok(0) => panic!("this process's resident size read as 0 bytes"),
        Ok(n) => n,
        Err(e) => panic!("cannot read this process's resident size: {e}"),
    }
}

/// Resident size in KiB.
pub fn kb() -> i64 {
    (bytes() / 1024) as i64
}

/// Resident size in MiB.
pub fn mb() -> u64 {
    bytes() / (1024 * 1024)
}

#[cfg(target_os = "linux")]
fn read() -> Result<u64, String> {
    let status = std::fs::read_to_string("/proc/self/status")
        .map_err(|e| format!("/proc/self/status: {e}"))?;
    let line = status
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .ok_or("/proc/self/status has no VmRSS line")?;
    let kb: u64 = line["VmRSS:".len()..]
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .map_err(|e| format!("`{line}`: {e}"))?;
    Ok(kb * 1024)
}

#[cfg(target_os = "macos")]
fn read() -> Result<u64, String> {
    /// `mach_task_basic_info`, which the kernel lays out 4-aligned.
    #[repr(C, packed(4))]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [i32; 2],
        system_time: [i32; 2],
        policy: i32,
        suspend_count: i32,
    }
    const MACH_TASK_BASIC_INFO: i32 = 20;
    unsafe extern "C" {
        static mach_task_self_: u32;
        fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
    }
    let mut info = std::mem::MaybeUninit::<MachTaskBasicInfo>::zeroed();
    // Counted in 4-byte words.
    let mut count = (std::mem::size_of::<MachTaskBasicInfo>() / 4) as u32;
    // SAFETY: `info` has room for `count` words, and the task port is
    // this process's own.
    let kr = unsafe {
        task_info(
            mach_task_self_,
            MACH_TASK_BASIC_INFO,
            info.as_mut_ptr().cast(),
            &mut count,
        )
    };
    if kr != 0 {
        return Err(format!("task_info returned {kr}"));
    }
    // SAFETY: filled by the successful call above.
    let info = unsafe { info.assume_init() };
    Ok(info.resident_size)
}

#[cfg(windows)]
fn read() -> Result<u64, String> {
    /// `PROCESS_MEMORY_COUNTERS`.
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
    // SAFETY: all-zero is a valid value of a struct of integers.
    let mut counters: ProcessMemoryCounters = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
    // SAFETY: the pseudo-handle names this process, and `counters` is
    // as large as `cb` says.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if ok == 0 {
        return Err(format!(
            "K32GetProcessMemoryInfo: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(counters.working_set_size as u64)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn read() -> Result<u64, String> {
    Err("no resident-size reader for this platform".into())
}
