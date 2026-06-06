//! System-linker + trampoline-stub plumbing for the AOT-via-object LLVM tier.
//!
//! Emits a position-independent shared object from a precompiled `.o` plus a
//! synthesised `stubs.o` that defines each runtime symbol as a real function
//! whose body is `mov <abs addr>, jmp <reg>`. `dlopen`s the result and
//! `dlsym`s exported function symbols.
//!
//! This is the V1 of the AOT JIT tier — required because MCJIT has cross-
//! thread MAP_JIT issues on Apple Silicon. Compiled code is loaded as a
//! genuine shared object instead of going through MCJIT's in-memory
//! mapper. See the design doc in commit message for details.
//!
//! The trampoline-stub mechanism replaces what MCJIT's
//! `add_global_mapping` does: bind a host-side function pointer (e.g.
//! `println`, `tensor_alloc`) to a named symbol the .so references. The
//! system linker has no notion of a "global mapping"; it needs the
//! symbol to exist in some object file. So we synthesise a tiny .o that
//! defines each runtime symbol as a globally-exported function whose
//! body is `movabsq $<host_ptr>, %rax ; jmpq *%rax` (x86_64) or
//! equivalent `movz`/`movk` + `br x16` (aarch64).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Target triple discriminator for arch-specific codegen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Triple {
    MacosAarch64,
    MacosX86_64,
    LinuxX86_64,
    LinuxAarch64,
    /// Any platform we don't have explicit trampoline-asm support for.
    /// Causes `find_linker` to return `NoLinker` so the runtime soft-
    /// falls back to the Cranelift tier.
    Unsupported,
}

impl Triple {
    /// Resolve the host triple via `cfg!` at compile time.
    pub const fn host() -> Self {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            Triple::MacosAarch64
        }
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        {
            Triple::MacosX86_64
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            Triple::LinuxX86_64
        }
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            Triple::LinuxAarch64
        }
        #[cfg(not(any(
            all(target_os = "macos", target_arch = "aarch64"),
            all(target_os = "macos", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "aarch64"),
        )))]
        {
            Triple::Unsupported
        }
    }

    /// File extension for the linker output.
    pub fn dylib_ext(&self) -> &'static str {
        match self {
            Triple::MacosAarch64 | Triple::MacosX86_64 => "dylib",
            Triple::LinuxX86_64 | Triple::LinuxAarch64 => "so",
            Triple::Unsupported => "so",
        }
    }
}

/// Errors from the link+load pipeline. The runtime treats `NoLinker`
/// as a soft-fail signal — the LLVM tier silently disables and
/// dispatch stays on Cranelift.
#[derive(Debug)]
pub enum LinkError {
    NoLinker,
    AssembleFailed(String),
    LinkFailed(String),
    DlOpenFailed(String),
    DlSymFailed { symbol: String, source: String },
    Io(std::io::Error),
    UnsupportedTriple,
}

impl std::fmt::Display for LinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkError::NoLinker => write!(
                f,
                "no system linker found (tried $ZYNTAX_LINKER, cc, clang, gcc)"
            ),
            LinkError::AssembleFailed(msg) => write!(f, "trampoline assemble failed: {msg}"),
            LinkError::LinkFailed(msg) => write!(f, "link failed: {msg}"),
            LinkError::DlOpenFailed(msg) => write!(f, "dlopen failed: {msg}"),
            LinkError::DlSymFailed { symbol, source } => {
                write!(f, "dlsym of '{symbol}' failed: {source}")
            }
            LinkError::Io(e) => write!(f, "io: {e}"),
            LinkError::UnsupportedTriple => write!(f, "unsupported host triple"),
        }
    }
}

impl std::error::Error for LinkError {}

impl From<std::io::Error> for LinkError {
    fn from(e: std::io::Error) -> Self {
        LinkError::Io(e)
    }
}

/// Find a system linker by probing `$ZYNTAX_LINKER`, then `cc`, `clang`,
/// `gcc` in that order. The first one whose `--version` exits 0 wins.
/// Result is intentionally NOT cached at module level — every install
/// re-probes so that env changes mid-process take effect (the cost is
/// ~tens of milliseconds, dwarfed by the linker run itself).
pub fn find_linker() -> Result<PathBuf, LinkError> {
    if let Ok(override_) = std::env::var("ZYNTAX_LINKER") {
        let candidate = PathBuf::from(&override_);
        if probe_linker(&candidate) {
            return Ok(candidate);
        }
    }
    for name in &["cc", "clang", "gcc"] {
        let candidate = PathBuf::from(name);
        if probe_linker(&candidate) {
            return Ok(candidate);
        }
    }
    Err(LinkError::NoLinker)
}

fn probe_linker(path: &Path) -> bool {
    Command::new(path)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Render the trampoline assembly for a given (triple, symbols) pair.
/// Each symbol becomes a globally-exported function whose body loads
/// the absolute host address into a scratch register and tail-jumps to
/// it. This lets the system linker resolve the .so's `call <name>`
/// against the stub; at runtime the stub's first instruction baked-in
/// the host pointer as an immediate.
///
/// The `(name, addr)` list comes from
/// [`LLVMJitBackend::runtime_symbols`](crate::llvm_jit_backend::LLVMJitBackend).
pub fn generate_trampoline_stubs(triple: Triple, symbols: &[(String, usize)]) -> String {
    let mut s = String::new();
    s.push_str(".text\n");
    for (name, addr) in symbols {
        let sym = match triple {
            Triple::MacosAarch64 | Triple::MacosX86_64 => format!("_{name}"),
            Triple::LinuxX86_64 | Triple::LinuxAarch64 => name.clone(),
            Triple::Unsupported => name.clone(),
        };
        match triple {
            Triple::MacosX86_64 => {
                s.push_str(&format!(".globl {sym}\n"));
                s.push_str(&format!("{sym}:\n"));
                s.push_str(&format!("    movabsq $0x{:x}, %rax\n", addr));
                s.push_str("    jmpq *%rax\n");
            }
            Triple::LinuxX86_64 => {
                s.push_str(&format!(".globl {sym}\n"));
                s.push_str(&format!(".type {sym},@function\n"));
                s.push_str(&format!("{sym}:\n"));
                s.push_str(&format!("    movabsq $0x{:x}, %rax\n", addr));
                s.push_str("    jmpq *%rax\n");
                s.push_str(&format!(".size {sym}, .-{sym}\n"));
            }
            Triple::MacosAarch64 => {
                let a = *addr as u64;
                s.push_str(&format!(".globl {sym}\n"));
                s.push_str(".p2align 2\n");
                s.push_str(&format!("{sym}:\n"));
                s.push_str(&format!("    movz x16, #0x{:x}\n", a & 0xFFFF));
                s.push_str(&format!(
                    "    movk x16, #0x{:x}, lsl #16\n",
                    (a >> 16) & 0xFFFF
                ));
                s.push_str(&format!(
                    "    movk x16, #0x{:x}, lsl #32\n",
                    (a >> 32) & 0xFFFF
                ));
                s.push_str(&format!(
                    "    movk x16, #0x{:x}, lsl #48\n",
                    (a >> 48) & 0xFFFF
                ));
                s.push_str("    br x16\n");
            }
            Triple::LinuxAarch64 => {
                let a = *addr as u64;
                s.push_str(&format!(".globl {sym}\n"));
                s.push_str(&format!(".type {sym},%function\n"));
                s.push_str(".p2align 2\n");
                s.push_str(&format!("{sym}:\n"));
                s.push_str(&format!("    movz x16, #0x{:x}\n", a & 0xFFFF));
                s.push_str(&format!(
                    "    movk x16, #0x{:x}, lsl #16\n",
                    (a >> 16) & 0xFFFF
                ));
                s.push_str(&format!(
                    "    movk x16, #0x{:x}, lsl #32\n",
                    (a >> 32) & 0xFFFF
                ));
                s.push_str(&format!(
                    "    movk x16, #0x{:x}, lsl #48\n",
                    (a >> 48) & 0xFFFF
                ));
                s.push_str("    br x16\n");
                s.push_str(&format!(".size {sym}, .-{sym}\n"));
            }
            Triple::Unsupported => {}
        }
    }
    s
}

/// Build a PIC shared library from `obj` + a synthesised trampoline `.o`,
/// linking in every runtime symbol. The .s file is written to `<obj>.stubs.s`,
/// assembled by the linker driver to `<obj>.stubs.o`, then linked with the
/// input `.o` to produce `dylib`. Temp files are cleaned on success.
pub fn link_to_dylib(
    obj: &Path,
    dylib: &Path,
    runtime_symbols: &[(String, usize)],
) -> Result<(), LinkError> {
    let triple = Triple::host();
    if matches!(triple, Triple::Unsupported) {
        return Err(LinkError::UnsupportedTriple);
    }

    let linker = find_linker()?;

    // Write the trampoline assembly file next to the input .o so cleanup
    // is colocated.
    let stubs_s = obj.with_extension("stubs.s");
    let stubs_o = obj.with_extension("stubs.o");
    let asm = generate_trampoline_stubs(triple, runtime_symbols);
    std::fs::write(&stubs_s, &asm)?;

    // Assemble: `linker -c stubs.s -o stubs.o`. The driver figures out
    // it's assembly from the extension.
    let assemble = Command::new(&linker)
        .args([
            "-c",
            stubs_s.to_str().unwrap_or(""),
            "-o",
            stubs_o.to_str().unwrap_or(""),
        ])
        .output()?;
    if !assemble.status.success() {
        let _ = std::fs::remove_file(&stubs_s);
        let stderr = String::from_utf8_lossy(&assemble.stderr).to_string();
        return Err(LinkError::AssembleFailed(stderr));
    }

    // Link to dylib. Per-platform flags. macOS: `-dynamiclib` plus
    // `-undefined dynamic_lookup` so memcpy/memset/etc. emitted by LLVM
    // lowering resolve against the host at dlopen time. Linux: `-shared`
    // + `-fPIC`.
    let mut args: Vec<String> = Vec::new();
    match triple {
        Triple::MacosAarch64 | Triple::MacosX86_64 => {
            args.push("-dynamiclib".to_string());
            args.push("-undefined".to_string());
            args.push("dynamic_lookup".to_string());
        }
        Triple::LinuxX86_64 | Triple::LinuxAarch64 => {
            args.push("-shared".to_string());
            args.push("-fPIC".to_string());
        }
        Triple::Unsupported => return Err(LinkError::UnsupportedTriple),
    }
    args.push("-o".to_string());
    args.push(dylib.to_string_lossy().into_owned());
    args.push(obj.to_string_lossy().into_owned());
    args.push(stubs_o.to_string_lossy().into_owned());

    let link = Command::new(&linker).args(&args).output()?;
    if !link.status.success() {
        let _ = std::fs::remove_file(&stubs_s);
        let _ = std::fs::remove_file(&stubs_o);
        let stderr = String::from_utf8_lossy(&link.stderr).to_string();
        return Err(LinkError::LinkFailed(stderr));
    }

    // Cleanup on success; failure to remove is non-fatal.
    let _ = std::fs::remove_file(&stubs_s);
    let _ = std::fs::remove_file(&stubs_o);
    Ok(())
}

/// `dlopen` the dylib and `dlsym` each requested symbol. Returns the
/// `Library` (must be stored by the caller to keep the mapping alive)
/// and a map of `name → host address as usize`.
///
/// Performs an fsync on the dylib first (to guarantee its inode is
/// visible to the loader on overlay-FS hosts) and a SeqCst fence
/// before the `dlopen` call.
#[cfg(unix)]
pub fn load_dylib(
    path: &Path,
    symbols: &[String],
) -> Result<(libloading::Library, HashMap<String, usize>), LinkError> {
    use std::fs::OpenOptions;
    use std::sync::atomic::{fence, Ordering};

    // Force the dylib's contents to disk and ensure metadata visibility.
    // Overlay-FS / tmpfs / NFS can leave a window where `close()` has
    // returned but `dlopen`'s subsequent `open() + mmap` sees an older
    // metadata view. Belt-and-braces; cheap.
    {
        let f = OpenOptions::new().read(true).open(path)?;
        f.sync_all()?;
    }
    fence(Ordering::SeqCst);

    // RTLD_NOW eagerly resolves every symbol — surfaces link errors
    // here instead of at first call. RTLD_GLOBAL makes the trampoline
    // symbols visible to the loaded library's own relocations.
    let lib_os = unsafe {
        libloading::os::unix::Library::open(Some(path), libc::RTLD_NOW | libc::RTLD_GLOBAL)
    }
    .map_err(|e| LinkError::DlOpenFailed(format!("{e}")))?;

    let mut map = HashMap::with_capacity(symbols.len());
    for name in symbols {
        let sym: libloading::os::unix::Symbol<*const u8> = unsafe { lib_os.get(name.as_bytes()) }
            .map_err(|e| LinkError::DlSymFailed {
            symbol: name.clone(),
            source: format!("{e}"),
        })?;
        let ptr = *sym as usize;
        map.insert(name.clone(), ptr);
    }

    // Convert OS-specific Library to the cross-platform Library so the
    // caller stores a single concrete type.
    let lib: libloading::Library = lib_os.into();
    Ok((lib, map))
}

#[cfg(not(unix))]
pub fn load_dylib(
    _path: &Path,
    _symbols: &[String],
) -> Result<(libloading::Library, HashMap<String, usize>), LinkError> {
    Err(LinkError::UnsupportedTriple)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_triple_resolves() {
        // Compile-time sanity check: host triple resolves to a known
        // arm on supported platforms.
        let t = Triple::host();
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert_eq!(t, Triple::MacosAarch64);
        }
        if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            assert_eq!(t, Triple::LinuxX86_64);
        }
    }

    #[test]
    fn dylib_ext_per_triple() {
        assert_eq!(Triple::MacosAarch64.dylib_ext(), "dylib");
        assert_eq!(Triple::MacosX86_64.dylib_ext(), "dylib");
        assert_eq!(Triple::LinuxX86_64.dylib_ext(), "so");
        assert_eq!(Triple::LinuxAarch64.dylib_ext(), "so");
    }

    #[test]
    fn trampoline_macos_x86_64_golden() {
        let syms = vec![("println".to_string(), 0xDEADBEEFCAFEBABE_usize)];
        let asm = generate_trampoline_stubs(Triple::MacosX86_64, &syms);
        // Mach-O underscore-prefix and absolute-load-then-jump.
        assert!(asm.contains(".globl _println"));
        assert!(asm.contains("_println:"));
        assert!(asm.contains("movabsq $0xdeadbeefcafebabe, %rax"));
        assert!(asm.contains("jmpq *%rax"));
    }

    #[test]
    fn trampoline_linux_x86_64_golden() {
        let syms = vec![("println".to_string(), 0x1122334455667788_usize)];
        let asm = generate_trampoline_stubs(Triple::LinuxX86_64, &syms);
        // ELF: no underscore prefix, has .type/@function and .size.
        assert!(asm.contains(".globl println"));
        assert!(!asm.contains(".globl _println"));
        assert!(asm.contains(".type println,@function"));
        assert!(asm.contains("println:"));
        assert!(asm.contains("movabsq $0x1122334455667788, %rax"));
        assert!(asm.contains("jmpq *%rax"));
        assert!(asm.contains(".size println, .-println"));
    }

    #[test]
    fn trampoline_macos_aarch64_golden() {
        // Address split into four 16-bit immediates: 0x1122_3344_5566_7788.
        let syms = vec![("foo".to_string(), 0x1122_3344_5566_7788_usize)];
        let asm = generate_trampoline_stubs(Triple::MacosAarch64, &syms);
        assert!(asm.contains(".globl _foo"));
        assert!(asm.contains(".p2align 2"));
        assert!(asm.contains("_foo:"));
        // bits 0..15 = 0x7788
        assert!(asm.contains("movz x16, #0x7788"));
        // bits 16..31 = 0x5566
        assert!(asm.contains("movk x16, #0x5566, lsl #16"));
        // bits 32..47 = 0x3344
        assert!(asm.contains("movk x16, #0x3344, lsl #32"));
        // bits 48..63 = 0x1122
        assert!(asm.contains("movk x16, #0x1122, lsl #48"));
        assert!(asm.contains("br x16"));
    }

    #[test]
    fn trampoline_linux_aarch64_golden() {
        let syms = vec![("bar".to_string(), 0xABCD_EF01_2345_6789_usize)];
        let asm = generate_trampoline_stubs(Triple::LinuxAarch64, &syms);
        assert!(asm.contains(".globl bar"));
        assert!(asm.contains(".type bar,%function"));
        assert!(asm.contains(".p2align 2"));
        assert!(asm.contains("bar:"));
        assert!(asm.contains("movz x16, #0x6789"));
        assert!(asm.contains("movk x16, #0x2345, lsl #16"));
        assert!(asm.contains("movk x16, #0xef01, lsl #32"));
        assert!(asm.contains("movk x16, #0xabcd, lsl #48"));
        assert!(asm.contains("br x16"));
        assert!(asm.contains(".size bar, .-bar"));
    }

    #[test]
    fn trampoline_multiple_symbols() {
        let syms = vec![
            ("a".to_string(), 0x1_usize),
            ("b".to_string(), 0x2_usize),
            ("c".to_string(), 0x3_usize),
        ];
        let asm = generate_trampoline_stubs(Triple::host(), &syms);
        // Header appears once.
        assert_eq!(asm.matches(".text\n").count(), 1);
    }

    #[test]
    fn find_linker_runs() {
        // On any developer host we should find at least one of
        // cc/clang/gcc. CI containers without a toolchain return
        // NoLinker which is a valid soft-fail. We can't assert Ok()
        // unconditionally — but the function must not panic.
        let _ = find_linker();
    }
}
