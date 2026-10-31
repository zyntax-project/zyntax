//! Machine code compiled in one process to run in another: a library's
//! functions compiled ahead of time and carried in its snapshot.
//!
//! Code this process compiles for itself holds run-time addresses as
//! constants: a callee's call cell, a bead's entry counter, a runtime
//! symbol. Portable code holds a relocation against a [`CodeAnchor`]
//! instead, and the process that places it binds each anchor to that
//! process's address. Functions and globals are named by their id in the
//! snapshot, before the shift every id takes when the snapshot is
//! decoded.

use serde::{Deserialize, Serialize};

/// A run-time address, or value, that portable code needs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodeAnchor {
    /// The call cell of the function.
    Cell(u32),
    /// The entry counter of the function's bead.
    EntryCounter(u32),
    /// The OSR helper slot of the function's loop header `site`.
    HelperSlot(u32, u64),
    /// The function's bead id, as a value.
    Bead(u32),
    /// The address of a runtime symbol, held as a value.
    Symbol(String),
    /// A runtime function the code calls, by its symbol, and by its id
    /// when the snapshot declares it: linked by name as the placing
    /// process links any import, unless the program defines a declared
    /// one there (a hook), and then the call reaches the program's
    /// function.
    Import(String, Option<u32>),
    /// The code of a function of the same snapshot.
    Function(u32),
    /// `offset` bytes into the code of a function of the same snapshot.
    FunctionOffset(u32, u32),
    /// A global of the same snapshot.
    Global(u32),
    /// A Cranelift libcall, by its name.
    LibCall(String),
}

impl CodeAnchor {
    /// Whether the anchor stands for an address or value the placing
    /// process knows at once, written into the code before it is
    /// installed, rather than code or data placed with it.
    pub fn is_value(&self) -> bool {
        matches!(
            self,
            Self::Cell(_)
                | Self::EntryCounter(_)
                | Self::HelperSlot(..)
                | Self::Bead(_)
                | Self::Symbol(_)
        )
    }
}

/// How a relocation is written, by the Cranelift kind it was emitted as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelocKind {
    Abs4,
    Abs8,
    X86PcRel4,
    X86CallPcRel4,
    X86CallPltRel4,
    X86GotPcRel4,
    Arm64Call,
    Aarch64AdrGotPage21,
    Aarch64Ld64GotLo12Nc,
    Aarch64AdrPrelPgHi21,
    Aarch64AddAbsLo12Nc,
}

#[cfg(feature = "cranelift-backend")]
impl RelocKind {
    /// The kind for a Cranelift relocation, or `None` for one portable
    /// code does not carry.
    pub fn of(reloc: cranelift_codegen::binemit::Reloc) -> Option<Self> {
        use cranelift_codegen::binemit::Reloc;
        Some(match reloc {
            Reloc::Abs4 => Self::Abs4,
            Reloc::Abs8 => Self::Abs8,
            Reloc::X86PCRel4 => Self::X86PcRel4,
            Reloc::X86CallPCRel4 => Self::X86CallPcRel4,
            Reloc::X86CallPLTRel4 => Self::X86CallPltRel4,
            Reloc::X86GOTPCRel4 => Self::X86GotPcRel4,
            Reloc::Arm64Call => Self::Arm64Call,
            Reloc::Aarch64AdrGotPage21 => Self::Aarch64AdrGotPage21,
            Reloc::Aarch64Ld64GotLo12Nc => Self::Aarch64Ld64GotLo12Nc,
            Reloc::Aarch64AdrPrelPgHi21 => Self::Aarch64AdrPrelPgHi21,
            Reloc::Aarch64AddAbsLo12Nc => Self::Aarch64AddAbsLo12Nc,
            _ => return None,
        })
    }

    /// The Cranelift relocation this kind was made from.
    pub fn reloc(self) -> cranelift_codegen::binemit::Reloc {
        use cranelift_codegen::binemit::Reloc;
        match self {
            Self::Abs4 => Reloc::Abs4,
            Self::Abs8 => Reloc::Abs8,
            Self::X86PcRel4 => Reloc::X86PCRel4,
            Self::X86CallPcRel4 => Reloc::X86CallPCRel4,
            Self::X86CallPltRel4 => Reloc::X86CallPLTRel4,
            Self::X86GotPcRel4 => Reloc::X86GOTPCRel4,
            Self::Arm64Call => Reloc::Arm64Call,
            Self::Aarch64AdrGotPage21 => Reloc::Aarch64AdrGotPage21,
            Self::Aarch64Ld64GotLo12Nc => Reloc::Aarch64Ld64GotLo12Nc,
            Self::Aarch64AdrPrelPgHi21 => Reloc::Aarch64AdrPrelPgHi21,
            Self::Aarch64AddAbsLo12Nc => Reloc::Aarch64AddAbsLo12Nc,
        }
    }
}

/// One place in a function's code that holds an anchor's address.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortableReloc {
    /// Bytes from the start of the function's code.
    pub offset: u32,
    pub kind: RelocKind,
    pub anchor: CodeAnchor,
    pub addend: i64,
}

/// One function's machine code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortableFunction {
    /// The function's id in the snapshot.
    pub id: u32,
    /// Its Cranelift signature as text, which the placing process's
    /// declaration of the function must match.
    pub signature: String,
    pub alignment: u64,
    pub code: Vec<u8>,
    pub relocs: Vec<PortableReloc>,
    /// Each runtime symbol whose registration shaped the code, with what
    /// the compile found for it: a plugin signature as text, or whether
    /// the symbol was registered at all (`present`, `absent`). The placing
    /// process must find the same.
    pub symbols: Vec<(String, String)>,
}

/// [`CodeAnchor`] read in place: its names borrowed from the image. The
/// variants are [`CodeAnchor`]'s, in its order, which the encoding names
/// them by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
pub enum CodeAnchorRef<'a> {
    Cell(u32),
    EntryCounter(u32),
    HelperSlot(u32, u64),
    Bead(u32),
    Symbol(&'a str),
    Import(&'a str, Option<u32>),
    Function(u32),
    FunctionOffset(u32, u32),
    Global(u32),
    LibCall(&'a str),
}

impl CodeAnchorRef<'_> {
    /// See [`CodeAnchor::is_value`].
    pub fn is_value(&self) -> bool {
        matches!(
            self,
            Self::Cell(_)
                | Self::EntryCounter(_)
                | Self::HelperSlot(..)
                | Self::Bead(_)
                | Self::Symbol(_)
        )
    }
}

/// [`PortableReloc`] read in place.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct PortableRelocRef<'a> {
    pub offset: u32,
    pub kind: RelocKind,
    #[serde(borrow)]
    pub anchor: CodeAnchorRef<'a>,
    pub addend: i64,
}

/// [`PortableFunction`] read in place: its code and names borrowed from
/// the image, so placing it allocates only its relocation list.
#[derive(Debug, Deserialize)]
pub struct PortableFunctionRef<'a> {
    pub id: u32,
    pub signature: &'a str,
    pub alignment: u64,
    pub code: &'a [u8],
    #[serde(borrow)]
    pub relocs: Vec<PortableRelocRef<'a>>,
    #[serde(borrow)]
    pub symbols: Vec<(&'a str, &'a str)>,
}

/// [`PortableCallee`] read in place.
#[derive(Debug, Deserialize)]
pub struct PortableCalleeRef<'a> {
    pub id: u32,
    pub name: &'a str,
    pub signature: &'a str,
}

/// What the code was compiled under, which the placing process must
/// match to use it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortableTarget {
    /// The target triple.
    pub triple: String,
    /// The architecture and operating system, which fix the instruction
    /// set and the calling convention: what the placing process's own
    /// target must share.
    pub platform: String,
    /// The compiler the code came from ([`crate::BUILD_ID`]).
    pub compiler: String,
    /// Whether loops carry OSR probes and entries count calls.
    pub probes: bool,
}

/// A snapshot's functions as machine code for one target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortableCode {
    pub target: PortableTarget,
    pub functions: Vec<PortableFunction>,
    /// Every function the code calls or takes the address of.
    pub callees: Vec<PortableCallee>,
    /// Every runtime function the code calls, with the signature it calls
    /// it with: the placing process declares an import it has not yet
    /// declared with this, so its own later calls agree with the code's.
    pub imports: Vec<PortableImport>,
}

impl PortableCode {
    /// The code as an image carries it, read by [`PortableImage`]: the
    /// target, an index of the functions and one of the callees by id,
    /// one of the imports in name order, then each function, callee and
    /// import encoded on its own.
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        let encoded = |e: postcard::Error| e.to_string();
        let mut blobs = Vec::new();
        let mut table = |entries: Vec<(u32, Vec<u8>)>, blobs: &mut Vec<u8>| {
            let mut out = Vec::with_capacity(4 + entries.len() * INDEX_ENTRY);
            out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
            for (id, bytes) in entries {
                for word in [id, blobs.len() as u32, bytes.len() as u32] {
                    out.extend_from_slice(&word.to_le_bytes());
                }
                blobs.extend_from_slice(&bytes);
            }
            out
        };
        let mut functions = self
            .functions
            .iter()
            .map(|f| Ok((f.id, postcard::to_allocvec(f).map_err(encoded)?)))
            .collect::<Result<Vec<_>, String>>()?;
        functions.sort_by_key(|(id, _)| *id);
        let mut callees = self
            .callees
            .iter()
            .map(|c| Ok((c.id, postcard::to_allocvec(c).map_err(encoded)?)))
            .collect::<Result<Vec<_>, String>>()?;
        callees.sort_by_key(|(id, _)| *id);
        let mut imports: Vec<&PortableImport> = self.imports.iter().collect();
        imports.sort_by(|a, b| a.name.cmp(&b.name));
        let imports = imports
            .into_iter()
            .enumerate()
            .map(|(at, i)| Ok((at as u32, postcard::to_allocvec(i).map_err(encoded)?)))
            .collect::<Result<Vec<_>, String>>()?;
        let functions = table(functions, &mut blobs);
        let callees = table(callees, &mut blobs);
        let imports = table(imports, &mut blobs);
        let header = postcard::to_allocvec(&self.target).map_err(encoded)?;
        let mut out =
            Vec::with_capacity(4 + header.len() + functions.len() + callees.len() + imports.len());
        out.extend_from_slice(&(header.len() as u32).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&functions);
        out.extend_from_slice(&callees);
        out.extend_from_slice(&imports);
        out.extend_from_slice(&blobs);
        Ok(out)
    }
}

/// Bytes of an id, a start and a length in a [`PortableImage`] index.
const INDEX_ENTRY: usize = 12;

/// The bytes a [`PortableImage`] reads: the executable's own, or a copy.
#[derive(Debug, Clone)]
pub enum ImageBytes {
    Static(&'static [u8]),
    Shared(std::sync::Arc<[u8]>),
}

impl std::ops::Deref for ImageBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Static(bytes) => bytes,
            Self::Shared(bytes) => bytes,
        }
    }
}

/// Encoded [`PortableCode`], read in place: nothing is decoded until it is
/// asked for, so a process pays for the functions it places and not for
/// the rest of the library.
#[derive(Debug, Clone)]
pub struct PortableImage {
    bytes: ImageBytes,
    header: std::ops::Range<usize>,
    functions: std::ops::Range<usize>,
    callees: std::ops::Range<usize>,
    imports: std::ops::Range<usize>,
    blobs: usize,
}

impl PortableImage {
    /// The image in `bytes`, checking only that its sections fit.
    pub fn new(bytes: ImageBytes) -> Result<Self, String> {
        let word = |at: usize| {
            bytes
                .get(at..at + 4)
                .map(|w| u32::from_le_bytes(w.try_into().expect("four bytes")) as usize)
                .ok_or_else(|| "portable code is truncated".to_string())
        };
        let header = 4..4 + word(0)?;
        let functions = header.end + 4..header.end + 4 + word(header.end)? * INDEX_ENTRY;
        let callees = functions.end + 4..functions.end + 4 + word(functions.end)? * INDEX_ENTRY;
        let imports = callees.end + 4..callees.end + 4 + word(callees.end)? * INDEX_ENTRY;
        if imports.end > bytes.len() {
            return Err("portable code is truncated".into());
        }
        Ok(Self {
            blobs: imports.end,
            bytes,
            header,
            functions,
            callees,
            imports,
        })
    }

    /// What the code was compiled under.
    pub fn target(&self) -> Result<PortableTarget, String> {
        postcard::from_bytes(&self.bytes[self.header.clone()]).map_err(|e| e.to_string())
    }

    /// The runtime function `name` the code calls, when it calls it. Only
    /// the names a search passes are read, and only the entry found is
    /// decoded.
    pub fn import(&self, name: &str) -> Option<Result<PortableImportRef<'_>, String>> {
        let (mut lo, mut hi) = (0, self.imports.len() / INDEX_ENTRY);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let [_, start, len] = self.entry(&self.imports, mid);
            let start = self.blobs + start as usize;
            let Some(bytes) = self.bytes.get(start..start + len as usize) else {
                return Some(Err("portable code is truncated".into()));
            };
            // An import is encoded name first.
            let at = match postcard::take_from_bytes::<&str>(bytes) {
                Ok((at, _)) => at,
                Err(e) => return Some(Err(e.to_string())),
            };
            match at.cmp(name) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    return Some(postcard::from_bytes(bytes).map_err(|e| e.to_string()));
                }
            }
        }
        None
    }

    /// The ids of the functions the image holds code for, in order.
    pub fn function_ids(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.functions.len() / INDEX_ENTRY).map(|i| self.entry(&self.functions, i)[0])
    }

    /// The code of function `id`, when the image holds it.
    pub fn function(&self, id: u32) -> Option<Result<PortableFunctionRef<'_>, String>> {
        self.decode(&self.functions, id)
    }

    /// The callee `id`, when the code calls it or takes its address.
    pub fn callee(&self, id: u32) -> Option<Result<PortableCalleeRef<'_>, String>> {
        self.decode(&self.callees, id)
    }

    fn entry(&self, table: &std::ops::Range<usize>, i: usize) -> [u32; 3] {
        let at = table.start + i * INDEX_ENTRY;
        let word = |k: usize| {
            u32::from_le_bytes(
                self.bytes[at + 4 * k..at + 4 * k + 4]
                    .try_into()
                    .expect("four bytes"),
            )
        };
        [word(0), word(1), word(2)]
    }

    fn decode<'a, T: serde::Deserialize<'a>>(
        &'a self,
        table: &std::ops::Range<usize>,
        id: u32,
    ) -> Option<Result<T, String>> {
        let (mut lo, mut hi) = (0, table.len() / INDEX_ENTRY);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let [at, start, len] = self.entry(table, mid);
            if at < id {
                lo = mid + 1;
            } else if at > id {
                hi = mid;
            } else {
                let start = self.blobs + start as usize;
                return Some(
                    self.bytes
                        .get(start..start + len as usize)
                        .ok_or_else(|| "portable code is truncated".to_string())
                        .and_then(|bytes| postcard::from_bytes(bytes).map_err(|e| e.to_string())),
                );
            }
        }
        None
    }
}

/// A runtime function portable code calls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortableImport {
    pub name: String,
    pub signature: PortableSignature,
}

/// A Cranelift signature in a form written into a snapshot and read back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortableSignature {
    pub params: Vec<PortableParam>,
    pub returns: Vec<PortableParam>,
    /// The calling convention as Cranelift names it.
    pub call_conv: String,
}

/// One parameter or return value of a [`PortableSignature`], each part
/// as Cranelift prints it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortableParam {
    pub value_type: String,
    /// `uext`, `sext`, or empty for none.
    pub extension: String,
    pub purpose: String,
}

#[cfg(feature = "cranelift-backend")]
impl PortableSignature {
    pub fn of(signature: &cranelift_codegen::ir::Signature) -> Self {
        use cranelift_codegen::ir::ArgumentExtension;
        let param = |p: &cranelift_codegen::ir::AbiParam| PortableParam {
            value_type: p.value_type.to_string(),
            extension: match p.extension {
                ArgumentExtension::None => "",
                ArgumentExtension::Uext => "uext",
                ArgumentExtension::Sext => "sext",
            }
            .to_string(),
            purpose: p.purpose.to_string(),
        };
        Self {
            params: signature.params.iter().map(param).collect(),
            returns: signature.returns.iter().map(param).collect(),
            call_conv: signature.call_conv.to_string(),
        }
    }

    /// The Cranelift signature, or `None` for a part this compiler does
    /// not know.
    pub fn signature(&self) -> Option<cranelift_codegen::ir::Signature> {
        fn part(p: &PortableParam) -> (&str, &str, &str) {
            (&p.value_type, &p.extension, &p.purpose)
        }
        read_signature(
            self.params.iter().map(part),
            self.returns.iter().map(part),
            &self.call_conv,
        )
    }
}

/// [`PortableSignature`] read in place.
#[derive(Debug, Deserialize)]
pub struct PortableSignatureRef<'a> {
    #[serde(borrow)]
    pub params: Vec<PortableParamRef<'a>>,
    #[serde(borrow)]
    pub returns: Vec<PortableParamRef<'a>>,
    pub call_conv: &'a str,
}

/// [`PortableParam`] read in place.
#[derive(Debug, Deserialize)]
pub struct PortableParamRef<'a> {
    pub value_type: &'a str,
    pub extension: &'a str,
    pub purpose: &'a str,
}

/// [`PortableImport`] read in place.
#[derive(Debug, Deserialize)]
pub struct PortableImportRef<'a> {
    pub name: &'a str,
    #[serde(borrow)]
    pub signature: PortableSignatureRef<'a>,
}

#[cfg(feature = "cranelift-backend")]
impl PortableSignatureRef<'_> {
    /// See [`PortableSignature::signature`].
    pub fn signature(&self) -> Option<cranelift_codegen::ir::Signature> {
        fn part<'a>(p: &PortableParamRef<'a>) -> (&'a str, &'a str, &'a str) {
            (p.value_type, p.extension, p.purpose)
        }
        read_signature(
            self.params.iter().map(part),
            self.returns.iter().map(part),
            self.call_conv,
        )
    }
}

/// A Cranelift signature from its parts as Cranelift prints them: each
/// parameter's value type, extension and purpose, and the calling
/// convention; `None` for a part this compiler does not know.
#[cfg(feature = "cranelift-backend")]
fn read_signature<'a>(
    params: impl Iterator<Item = (&'a str, &'a str, &'a str)>,
    returns: impl Iterator<Item = (&'a str, &'a str, &'a str)>,
    call_conv: &str,
) -> Option<cranelift_codegen::ir::Signature> {
    use cranelift_codegen::ir::{AbiParam, ArgumentExtension, types};
    let value_type = |name: &str| {
        let (lane, lanes) = match name.split_once('x') {
            Some((lane, lanes)) => (lane, lanes.parse::<u32>().ok()?),
            None => (name, 1),
        };
        // The names Cranelift prints its scalar types by.
        let lane = match lane {
            "i8" => types::I8,
            "i16" => types::I16,
            "i32" => types::I32,
            "i64" => types::I64,
            "i128" => types::I128,
            "f16" => types::F16,
            "f32" => types::F32,
            "f64" => types::F64,
            "f128" => types::F128,
            _ => return None,
        };
        if lanes == 1 {
            Some(lane)
        } else {
            lane.by(lanes)
        }
    };
    let param = |(value, extension, purpose): (&str, &str, &str)| {
        let extension = match extension {
            "" => ArgumentExtension::None,
            "uext" => ArgumentExtension::Uext,
            "sext" => ArgumentExtension::Sext,
            _ => return None,
        };
        Some(AbiParam {
            value_type: value_type(value)?,
            purpose: purpose.parse().ok()?,
            extension,
        })
    };
    Some(cranelift_codegen::ir::Signature {
        params: params.map(param).collect::<Option<_>>()?,
        returns: returns.map(param).collect::<Option<_>>()?,
        call_conv: call_conv.parse().ok()?,
    })
}

/// A function portable code calls or takes the address of.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortableCallee {
    /// Its id in the snapshot.
    pub id: u32,
    pub name: String,
    /// Its Cranelift signature as text: the code calls it with this
    /// signature, so the placing process's declaration must match.
    pub signature: String,
}

/// Compile every function of the snapshot module `source` for `triple`,
/// each body finished as the runtime finishes a snapshot body before its
/// first compile (see `TieredBackend`'s lazy compiler). `probes` is
/// whether loops carry OSR probes, as the runtime that places the code
/// emits them; `symbols` are the runtime symbols it registers, whose
/// signatures shape calls to them.
#[cfg(feature = "cranelift-backend")]
pub fn compile_snapshot_module(
    source: &crate::bytecode::LazyModule,
    triple: &str,
    probes: bool,
    symbols: &[crate::zrtl::RuntimeSymbolInfo],
) -> crate::CompilerResult<PortableCode> {
    use crate::hir::HirId;
    crate::boxes::set_interning(true);
    let mut module = source.shell().clone();
    let externs = crate::boxes::externs_of(&module);
    let pure_fns: crate::hir::IdHashSet<HirId> = module
        .functions
        .iter()
        .filter(|(_, f)| f.signature.is_pure)
        .map(|(id, _)| *id)
        .collect();
    for (id, mut f) in source.functions() {
        if f.is_external {
            continue;
        }
        f.attributes.deferred = false;
        let boxed = crate::boxes::run_function(&mut f, &externs);
        if boxed.expanded + boxed.made + boxed.released + boxed.shared > 0 {
            crate::licm::run(&mut f);
            crate::cse::eliminate_with(&mut f, &pure_fns);
        }
        module.functions.insert(id, f);
    }
    let mut backend =
        crate::cranelift_backend::CraneliftBackend::portable_for_target(triple, source.id_base())?;
    backend.set_emit_osr_probes(probes);
    backend.register_runtime_symbols(symbols.iter().map(|s| (s.name, s.ptr)));
    backend.register_symbol_signatures(symbols);
    backend.compile_module(&module)?;
    let target = PortableTarget {
        triple: triple.to_string(),
        platform: backend.platform(),
        compiler: crate::BUILD_ID.to_string(),
        probes,
    };
    Ok(backend
        .take_portable_code(target)
        .expect("a portable backend writes portable code"))
}

/// Whether this compiler can compile for `triple`: its Cranelift carries
/// only the instruction sets its build selected.
#[cfg(feature = "cranelift-backend")]
pub fn compiles_for(triple: &str) -> bool {
    cranelift_codegen::isa::lookup_by_name(triple).is_ok()
}

/// The triple of the machine this process runs on, as Cranelift names it.
#[cfg(feature = "cranelift-backend")]
pub fn host_triple() -> String {
    cranelift_native::builder()
        .map(|b| b.triple().to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn function(id: u32) -> PortableFunction {
        let anchors = [
            CodeAnchor::Cell(1),
            CodeAnchor::EntryCounter(2),
            CodeAnchor::HelperSlot(3, 99),
            CodeAnchor::Bead(4),
            CodeAnchor::Symbol("sym".into()),
            CodeAnchor::Import("imp".into(), Some(5)),
            CodeAnchor::Import("bare".into(), None),
            CodeAnchor::Function(6),
            CodeAnchor::FunctionOffset(7, 8),
            CodeAnchor::Global(9),
            CodeAnchor::LibCall("FloorF64".into()),
        ];
        PortableFunction {
            id,
            signature: "(i64) -> i64 system_v".into(),
            alignment: 16,
            code: (0..40).collect(),
            relocs: anchors
                .into_iter()
                .enumerate()
                .map(|(i, anchor)| PortableReloc {
                    offset: i as u32 * 4,
                    kind: RelocKind::Abs8,
                    anchor,
                    addend: -(i as i64),
                })
                .collect(),
            symbols: vec![("zl_x".into(), "present".into())],
        }
    }

    #[test]
    fn functions_read_in_place_as_they_were_written() {
        let code = PortableCode {
            target: PortableTarget {
                triple: "t".into(),
                platform: "p".into(),
                compiler: "c".into(),
                probes: true,
            },
            functions: vec![function(3), function(1)],
            callees: vec![PortableCallee {
                id: 6,
                name: "callee".into(),
                signature: "() system_v".into(),
            }],
            imports: vec![PortableImport {
                name: "imp".into(),
                signature: PortableSignature {
                    params: vec![],
                    returns: vec![],
                    call_conv: "system_v".into(),
                },
            }],
        };
        let image = PortableImage::new(ImageBytes::Shared(code.encode().unwrap().into())).unwrap();
        let written = function(1);
        let read = image.function(1).unwrap().unwrap();
        assert_eq!(read.id, written.id);
        assert_eq!(read.signature, written.signature);
        assert_eq!(read.alignment, written.alignment);
        assert_eq!(read.code, &written.code[..]);
        assert_eq!(read.symbols, vec![("zl_x", "present")]);
        assert_eq!(read.relocs.len(), written.relocs.len());
        for (r, w) in read.relocs.iter().zip(&written.relocs) {
            assert_eq!((r.offset, r.kind, r.addend), (w.offset, w.kind, w.addend));
            let same = match (&r.anchor, &w.anchor) {
                (CodeAnchorRef::Cell(a), CodeAnchor::Cell(b))
                | (CodeAnchorRef::EntryCounter(a), CodeAnchor::EntryCounter(b))
                | (CodeAnchorRef::Bead(a), CodeAnchor::Bead(b))
                | (CodeAnchorRef::Function(a), CodeAnchor::Function(b))
                | (CodeAnchorRef::Global(a), CodeAnchor::Global(b)) => a == b,
                (CodeAnchorRef::HelperSlot(a, s), CodeAnchor::HelperSlot(b, t)) => a == b && s == t,
                (CodeAnchorRef::FunctionOffset(a, s), CodeAnchor::FunctionOffset(b, t)) => {
                    a == b && s == t
                }
                (CodeAnchorRef::Symbol(a), CodeAnchor::Symbol(b))
                | (CodeAnchorRef::LibCall(a), CodeAnchor::LibCall(b)) => *a == b,
                (CodeAnchorRef::Import(a, s), CodeAnchor::Import(b, t)) => *a == b && s == t,
                _ => false,
            };
            assert!(same, "{:?} read as {:?}", w.anchor, r.anchor);
            assert_eq!(r.anchor.is_value(), w.anchor.is_value());
        }
        let callee = image.callee(6).unwrap().unwrap();
        assert_eq!(
            (callee.id, callee.name, callee.signature),
            (6, "callee", "() system_v")
        );
        assert!(image.import("imp").unwrap().is_ok());
        assert!(image.import("missing").is_none());
        assert!(image.function(2).is_none());
    }
}

#[cfg(all(test, feature = "cranelift-backend"))]
mod signature_tests {
    use super::*;
    use cranelift_codegen::ir::{AbiParam, ArgumentExtension, Signature, types};
    use cranelift_codegen::isa::CallConv;

    #[test]
    fn signatures_read_back_as_written() {
        let mut sig = Signature::new(CallConv::SystemV);
        for ty in [
            types::I8,
            types::I16,
            types::I32,
            types::I64,
            types::I128,
            types::F16,
            types::F32,
            types::F64,
            types::F128,
            types::I32X4,
            types::F64X2,
        ] {
            sig.params.push(AbiParam::new(ty));
        }
        sig.params.push(AbiParam::new(types::I8).uext());
        sig.params[0].extension = ArgumentExtension::Sext;
        sig.returns.push(AbiParam::new(types::F64));
        assert_eq!(PortableSignature::of(&sig).signature(), Some(sig));
    }
}
