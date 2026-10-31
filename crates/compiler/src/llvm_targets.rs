//! The LLVM targets this build links: the host's, and no other.
//!
//! llvm-sys is built with `disable-alltargets-init`, so it compiles
//! none of its C wrappers around `llvm-c/Target.h`. Those wrappers sit
//! in one object, and its `LLVM_InitializeAll*` functions name every
//! target LLVM was configured with, so linking any one wrapper linked
//! every target and ran all their static initialisers at each process
//! start. inkwell still calls the wrappers by name (an execution
//! engine initialises the native target itself), so they are defined
//! here. Each `All` wrapper initialises the host target, which is every
//! target this build links.

use inkwell::llvm_sys::prelude::LLVMBool;

#[cfg(target_arch = "aarch64")]
mod host {
    pub use inkwell::llvm_sys::target::{
        LLVMInitializeAArch64AsmParser as asm_parser,
        LLVMInitializeAArch64AsmPrinter as asm_printer,
        LLVMInitializeAArch64Disassembler as disassembler, LLVMInitializeAArch64Target as target,
        LLVMInitializeAArch64TargetInfo as info, LLVMInitializeAArch64TargetMC as mc,
    };
}

#[cfg(target_arch = "x86_64")]
mod host {
    pub use inkwell::llvm_sys::target::{
        LLVMInitializeX86AsmParser as asm_parser, LLVMInitializeX86AsmPrinter as asm_printer,
        LLVMInitializeX86Disassembler as disassembler, LLVMInitializeX86Target as target,
        LLVMInitializeX86TargetInfo as info, LLVMInitializeX86TargetMC as mc,
    };
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
compile_error!("the LLVM backend links the host's LLVM target, and has none for this architecture");

/// Register the host target with LLVM, with its assembly printer,
/// parser and disassembler: everything an execution engine or an
/// object emitter here needs. Safe to call more than once. Calling the
/// wrappers from here keeps them linked whatever order a linker reads
/// the archives in; a binary that drives inkwell itself calls this
/// first, or it does not link.
pub fn initialize_host() {
    LLVM_InitializeNativeTarget();
    LLVM_InitializeNativeAsmPrinter();
    LLVM_InitializeNativeAsmParser();
    LLVM_InitializeNativeDisassembler();
}

/// Run the image's static initialisers, once, when its link left them to
/// the runtime (`zrtl_native::build::emit_deferred_initializers`); they
/// are LLVM's, and nothing here touches LLVM before they have run. Does
/// nothing for an image whose loader ran them, as it does for one linked
/// without that step.
pub fn run_static_initializers() {
    static RAN: std::sync::Once = std::sync::Once::new();
    RAN.call_once(|| {
        #[cfg(target_os = "macos")]
        // SAFETY: each offset names an initialiser of this image that the
        // loader skipped, run once, in the order the loader would.
        unsafe {
            macho::run_deferred_initializers()
        };
        #[cfg(target_os = "linux")]
        // SAFETY: each pointer names an initialiser of this executable
        // that the loader skipped, run once, in the order it would.
        unsafe {
            elf::run_deferred_initializers()
        };
    });
}

#[cfg(target_os = "linux")]
mod elf {
    use std::os::unix::fs::FileExt;

    /// The section the link moved LLVM's initialiser pointers to
    /// (`zrtl_native::build::DEFERRED_INITIALIZERS_SECTION`).
    const SECTION: &[u8] = b".zyntax_llvm_init";

    /// Call each pointer of this executable's deferred-initialiser
    /// section, found through the section headers of the executable
    /// file, which the loader does not map. Nothing when the section is
    /// absent, as in an executable linked without that step.
    pub(super) unsafe fn run_deferred_initializers() {
        let Some((addr, size, vaddr_of_phdr)) = locate() else {
            return;
        };
        // SAFETY: getauxval has no preconditions.
        let phdr = unsafe { libc::getauxval(libc::AT_PHDR) } as usize;
        let bias = phdr.wrapping_sub(vaddr_of_phdr);
        let first = addr.wrapping_add(bias) as *const usize;
        for i in 0..size / std::mem::size_of::<usize>() {
            // SAFETY: the loader relocated each pointer of the section to
            // an initialiser of this executable that it did not run.
            unsafe {
                let init: extern "C" fn() = std::mem::transmute(*first.add(i));
                init();
            }
        }
    }

    /// The section's address and size, and the address the program
    /// headers are linked at, read from the executable file.
    fn locate() -> Option<(usize, usize, usize)> {
        let file = std::fs::File::open("/proc/self/exe").ok()?;
        let read = |at: u64, len: usize| -> Option<Vec<u8>> {
            let mut buf = vec![0; len];
            file.read_exact_at(&mut buf, at).ok()?;
            Some(buf)
        };
        let header = read(0, 64)?;
        // ELFCLASS64, ELFDATA2LSB: the only images linked to defer.
        if header.get(..6)? != b"\x7fELF\x02\x01" {
            return None;
        }
        let u16_at =
            |b: &[u8], at: usize| Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?));
        let u32_at =
            |b: &[u8], at: usize| Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?));
        let u64_at =
            |b: &[u8], at: usize| Some(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?));
        let phoff = u64_at(&header, 0x20)?;
        let shoff = u64_at(&header, 0x28)?;
        let phentsize = u16_at(&header, 0x36)? as usize;
        let phnum = u16_at(&header, 0x38)? as usize;
        let shentsize = u16_at(&header, 0x3a)? as usize;
        let shnum = u16_at(&header, 0x3c)? as usize;
        let shstrndx = u16_at(&header, 0x3e)? as usize;
        let programs = read(phoff, phentsize * phnum)?;
        const PT_LOAD: u32 = 1;
        const PT_PHDR: u32 = 6;
        let program = |i: usize| i * phentsize;
        let phdr_vaddr = (0..phnum)
            .find(|&i| u32_at(&programs, program(i)) == Some(PT_PHDR))
            .and_then(|i| u64_at(&programs, program(i) + 0x10))
            .or_else(|| {
                (0..phnum)
                    .find(|&i| {
                        u32_at(&programs, program(i)) == Some(PT_LOAD)
                            && u64_at(&programs, program(i) + 0x08) == Some(0)
                    })
                    .and_then(|i| Some(u64_at(&programs, program(i) + 0x10)? + phoff))
            })?;
        let sections = read(shoff, shentsize * shnum)?;
        let section = |i: usize| i * shentsize;
        let names_at = u64_at(&sections, section(shstrndx) + 0x18)?;
        let names_len = u64_at(&sections, section(shstrndx) + 0x20)? as usize;
        let names = read(names_at, names_len)?;
        (0..shnum).find_map(|i| {
            let name = u32_at(&sections, section(i))? as usize;
            let end = name + names.get(name..)?.iter().position(|&c| c == 0)?;
            if names.get(name..end)? != SECTION {
                return None;
            }
            Some((
                u64_at(&sections, section(i) + 0x10)? as usize,
                u64_at(&sections, section(i) + 0x20)? as usize,
                phdr_vaddr as usize,
            ))
        })
    }
}

#[cfg(target_os = "macos")]
mod macho {
    const LC_SEGMENT_64: u32 = 0x19;
    const SECTION_TYPE: u32 = 0xff;
    const S_REGULAR: u32 = 0x0;

    #[repr(C)]
    struct Header {
        magic: u32,
        cputype: i32,
        cpusubtype: i32,
        filetype: u32,
        ncmds: u32,
        sizeofcmds: u32,
        flags: u32,
        reserved: u32,
    }

    #[repr(C)]
    struct Command {
        cmd: u32,
        cmdsize: u32,
    }

    #[repr(C)]
    struct Segment {
        cmd: u32,
        cmdsize: u32,
        segname: [u8; 16],
        vmaddr: u64,
        vmsize: u64,
        fileoff: u64,
        filesize: u64,
        maxprot: i32,
        initprot: i32,
        nsects: u32,
        flags: u32,
    }

    #[repr(C)]
    struct Section {
        sectname: [u8; 16],
        segname: [u8; 16],
        addr: u64,
        size: u64,
        offset: u32,
        align: u32,
        reloff: u32,
        nreloc: u32,
        flags: u32,
        reserved: [u32; 3],
    }

    fn named(field: &[u8; 16], name: &[u8]) -> bool {
        field.split(|&b| b == 0).next() == Some(name)
    }

    /// Call each initialiser `__TEXT,__init_offsets` names when the
    /// section is typed as regular data, which the loader skips: offsets
    /// from the image's header, run in order.
    pub(super) unsafe fn run_deferred_initializers() {
        // SAFETY: the image's own header and load commands, mapped for
        // as long as the image is.
        unsafe {
            let mut info: libc::Dl_info = std::mem::zeroed();
            if libc::dladdr(run_deferred_initializers as *const libc::c_void, &mut info) == 0 {
                return;
            }
            let base = info.dli_fbase as usize;
            let header = &*(base as *const Header);
            let mut at = base + std::mem::size_of::<Header>();
            let mut slide = None;
            let mut offsets = None;
            for _ in 0..header.ncmds {
                let command = &*(at as *const Command);
                if command.cmd == LC_SEGMENT_64 {
                    let segment = &*(at as *const Segment);
                    if named(&segment.segname, b"__TEXT") {
                        slide = Some(base.wrapping_sub(segment.vmaddr as usize));
                        let sections = (at + std::mem::size_of::<Segment>()) as *const Section;
                        for i in 0..segment.nsects as usize {
                            let section = &*sections.add(i);
                            if named(&section.sectname, b"__init_offsets")
                                && section.flags & SECTION_TYPE == S_REGULAR
                            {
                                offsets = Some((section.addr, section.size));
                            }
                        }
                    }
                }
                at += command.cmdsize as usize;
            }
            let (Some(slide), Some((addr, size))) = (slide, offsets) else {
                return;
            };
            let first = (addr as usize).wrapping_add(slide) as *const u32;
            for i in 0..size as usize / 4 {
                let init: extern "C" fn() = std::mem::transmute(base + *first.add(i) as usize);
                init();
            }
        }
    }
}

/// Native and All wrappers share one process-wide registration.
fn initialize() {
    static INITIALIZED: std::sync::Once = std::sync::Once::new();
    INITIALIZED.call_once(|| {
        run_static_initializers();
        // SAFETY: registration finishes once before any caller proceeds.
        unsafe {
            host::info();
            host::target();
            host::mc();
            host::asm_parser();
            host::asm_printer();
            host::disassembler();
        }
    });
}

/// `LLVMInitializeNativeTarget`: the host's target info, target and
/// machine-code layer. Returns 0, LLVM's success.
#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeNativeTarget() -> LLVMBool {
    initialize();
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeNativeAsmParser() -> LLVMBool {
    initialize();
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeNativeAsmPrinter() -> LLVMBool {
    initialize();
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeNativeDisassembler() -> LLVMBool {
    initialize();
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeAllTargetInfos() {
    initialize();
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeAllTargets() {
    initialize();
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeAllTargetMCs() {
    initialize();
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeAllAsmPrinters() {
    initialize();
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeAllAsmParsers() {
    initialize();
}

#[unsafe(no_mangle)]
pub extern "C" fn LLVM_InitializeAllDisassemblers() {
    initialize();
}
