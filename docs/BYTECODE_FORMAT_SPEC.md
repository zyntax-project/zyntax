# Zyntax Bytecode Format Specification

**Version**: 3.0
**Status**: Stable
**Last Updated**: 2026-09-25

---

## Overview

A Zyntax bytecode file (`.zbc`) holds one HIR module (`HirModule`): a fixed
44-byte header followed by a payload. The header is a hand-written
little-endian record. The payload is the serde encoding of the HIR data model
in one of four encodings, chosen by the header's format byte.

The format has no sections of its own: no string table, type table or
constant pool, and no debug or metadata section. Names are stored inline as
strings, types as `HirType` values and constants as `HirConstant` values,
wherever the data model places them. The metadata a module carries is part
of the data model: `HirModule.version` and `dependencies`, and source `Span`s
on values.

| What | Where |
|------|-------|
| Header, encodings, checks | `crates/compiler/src/bytecode.rs` |
| Payload data model | `crates/compiler/src/hir.rs`, plus `Span`, `TypeId`, `InternedString` and `ParamOwnership` from `crates/typed_ast` |
| CLI loader | `crates/zyntax_cli/src/formats/hir_bytecode.rs` |

The payload layout is defined by the codec applied to the declarations in
those files. Section 5 transcribes them for format 3.0; where the two
disagree, the source is authoritative.

### Where the format is used

- `zyntax compile <file>.zbc` reads one module, selected by the `.zbc`
  extension or `--format hir-bytecode`. A directory argument is searched
  recursively for `.zbc` files; more than one file is refused.
- ZPack archives store each module at `modules/<path>.zbc`, in the Postcard
  encoding.
- Language snapshots embed each lowered module as a Split image (format 3).

---

## 1. File Layout

```
offset 0    header   44 bytes
offset 44   payload  every remaining byte of the file
```

The payload always runs to the end of the file. Nothing may follow it: the
checksum covers every byte after the header.

---

## 2. Header

All multi-byte fields are little-endian.

| Offset | Size | Field | Written as | Checked on read |
|--------|------|-------|------------|-----------------|
| 0x00 | 4 | `magic` | u32 `0x5A424300`, bytes `00 43 42 5A` | must match |
| 0x04 | 2 | `major_version` | u16 `3` | must equal 3 |
| 0x06 | 2 | `minor_version` | u16 `0` | not checked |
| 0x08 | 1 | `format` | u8 payload encoding, section 3 | must be 0 to 3 |
| 0x09 | 3 | padding | zero | not checked |
| 0x0C | 4 | `flags` | u32 `0` | not checked |
| 0x10 | 16 | `module_id` | the module's `HirId` as u32, then 12 zero bytes | not checked |
| 0x20 | 8 | `payload_size` | u64 payload length in bytes | not checked |
| 0x28 | 4 | `checksum` | u32 CRC-32 of the payload | must match, except on a trusted read |

**Magic.** The constant is `0x5A424300` written little-endian, so a file
starts with the bytes `00 43 42 5A`. Writing the ASCII bytes `Z B C \0` gives
a different u32 and the file is refused.

**Flags.** No bits are defined. Writers write 0; readers ignore the field.

**Module id.** The first four bytes are the module's `HirId` as the writer
numbered it, before relocation (section 5.1). Only `bytecode_stats` reports
it; decoding takes the module id from the payload.

**Payload size.** Informational. Readers take the payload as bytes 44 to the
end of the file whatever this field says.

**Checksum.** CRC-32 as computed by `crc32fast` and zlib (reflected
polynomial `0xEDB88320`, initial value `0xFFFFFFFF`, final XOR `0xFFFFFFFF`)
over bytes 44 to the end of the file. The header is not covered.

### 2.1 What a reader checks

`deserialize_module` checks, in order:

1. The file is at least 44 bytes, else `InvalidFormat`.
2. `magic` matches, else `InvalidFormat`.
3. `major_version` is 3, else `VersionMismatch`. The minor version is not
   compared.
4. The CRC-32 of bytes 44 to the end matches `checksum`, else
   `ChecksumMismatch`.
5. `format` is 0 to 3, else `InvalidFormat`.
6. The payload decodes with that encoding, else `DeserializationError`. A
   Split image that fails its structural checks (section 6) gives
   `InvalidFormat`.

The other entry points differ only as follows:

| Entry point | Difference |
|-------------|------------|
| `deserialize_module_lazy` | A Split image stays encoded and decodes one function at a time; any other format decodes whole. |
| `deserialize_module_lazy_trusted` | Skips step 4 for a Split image. For images the running build embedded in its own executable. |
| `bytecode_stats` | Steps 1, 2, 3 and 5; no checksum, no payload decode. |

The reader does not check that the decoded HIR is well formed (section 5.4).

---

## 3. Payload Encodings

| `format` | Name | Codec | Used by |
|----------|------|-------|---------|
| 0 | Postcard | `postcard` 1.x, `to_allocvec` / `from_bytes` | ZPack archives, the ZynML benchmark HIR cache |
| 1 | JSON | `serde_json`, `to_vec_pretty` / `from_slice` | API only |
| 2 | Bincode | `bincode` 1.3 crate-root `serialize` / `deserialize` | API only |
| 3 | Split | Postcard, each function encoded on its own (section 6) | language snapshots |

All four encode the same data model (section 5). Formats 0 and 2 ignore
bytes after the encoded module, format 3 refuses them (section 6.2), and
JSON refuses anything but whitespace after it. A producer emits nothing after
the module in any encoding.

### 3.1 Postcard (formats 0 and 3)

Postcard writes no field names, no type tags and no field counts. A value is
its fields in declaration order, encoded by these rules:

| serde type | Encoding |
|------------|----------|
| `bool` | 1 byte, `00` or `01`; any other byte is refused |
| `u8` | 1 byte |
| `i8` | 1 byte, two's complement |
| `u16`, `u32`, `u64`, `u128` | unsigned LEB128 varint: 7 bits per byte, low group first, high bit set on every byte but the last; at most 3, 5, 10, 19 bytes; more bytes than that, or a last byte with bits beyond the type's width, is refused |
| `i16`, `i32`, `i64`, `i128` | zigzag (`(n << 1) ^ (n >> (bits - 1))`), then as the unsigned type |
| `usize` | as `u64` |
| `f32`, `f64` | IEEE 754 bits, 4 or 8 bytes little-endian |
| `String`, `InternedString` | varint byte length, then UTF-8 bytes; invalid UTF-8 is refused |
| `Option<T>` | `00` for `None`; `01` then `T` for `Some` |
| struct, tuple, tuple struct | each field in declaration order, nothing else |
| newtype struct (`HirId`, `TypeId`, `LifetimeId`) | the inner value |
| `Box<T>` | `T` |
| `Vec<T>`, `HashSet<T>` | varint element count, then the elements |
| `IndexMap<K, V>` | varint entry count, then key, value, key, value, ... |
| enum | varint variant index (0-based declaration order), then the variant's fields as a struct or tuple; a unit variant is the index alone |

A field marked `#[serde(default)]` in the source is still required in
Postcard: fields are positional and none may be left out.

`IndexMap` entries keep their order, which is the module's order.
`HashSet` elements come in the writer's hash iteration order, so two
encodings of the same module can differ byte for byte. Readers do not depend
on set order.

### 3.2 JSON (format 1)

serde's default JSON representation:

- A struct is an object keyed by field name. Unknown fields are ignored.
  Fields marked `#[serde(default)]` may be omitted.
- An enum is externally tagged: a unit variant is a string (`"I32"`); any
  other variant is a one-key object, `{"Ptr": "I8"}`,
  `{"Array": ["I32", 4]}`, `{"Load": {"result": 7, ...}}`.
- A map keyed by `HirId` or `TypeId` is an object whose keys are the decimal
  ids as strings (`"12"`).
- `HirId`, `TypeId` and `LifetimeId` are JSON numbers; `InternedString` is a
  JSON string; `None` is `null`.
- A non-finite `f32` or `f64` is written as `null` and refused on read, so a
  module holding a NaN or infinite constant cannot be carried as JSON.

### 3.3 Bincode (format 2)

The bincode 1.3 crate-root configuration: little-endian, fixed-width
integers, trailing bytes allowed, no size limit. Integers take their full
width (`usize` as 8 bytes). String, sequence and map lengths are u64.
`Option` is a `00`/`01` byte. An enum variant index is a u32. Field order is
as in Postcard.

---

## 4. Examples

### 4.1 A complete file

An empty module with id 1 named `m`, in the Postcard encoding (57 bytes):

```
0000  00 43 42 5a 03 00 00 00 00 00 00 00 00 00 00 00
0010  01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0020  0d 00 00 00 00 00 00 00 57 08 07 d2 01 01 6d 00
0030  00 00 00 00 00 00 00 00 00
```

| Bytes | Meaning |
|-------|---------|
| `00 43 42 5a` | magic |
| `03 00` `00 00` | version 3.0 |
| `00` `00 00 00` | format 0 (Postcard), padding |
| `00 00 00 00` | flags |
| `01 00 00 00` + 12 zero bytes | module_id: module id 1 |
| `0d 00 00 00 00 00 00 00` | payload_size 13 |
| `57 08 07 d2` | checksum `0xD2070857` |
| `01` | `HirModule.id` = 1 |
| `01 6d` | `name` = `"m"` |
| `00` x 10 | `functions`, `globals`, `types`, `imports`, `exports` empty; `version` 0; `dependencies`, `effects`, `handlers` empty; `automatic_release` false |

### 4.2 Instructions and terminators

`Binary { op: Add, result: 5, ty: I32, left: 3, right: 4 }` in Postcard:

```
00  HirInstruction variant 0 (Binary)
00  op: BinaryOp variant 0 (Add)
05  result: HirId 5
04  ty: HirType variant 4 (I32)
03  left: HirId 3
04  right: HirId 4
```

`Return { values: [5] }` is `00 01 05`: variant 0, one element, HirId 5.

---

## 5. Payload Data Model

This is the format 3.0 schema. Types are written in Rust notation; every
struct field is present, in the order shown. A `Box<T>` in the source is
shown as `T`, which encodes identically.

### 5.1 Identifiers and scalars

**`HirId`** is a u32. `0` is a sentinel meaning "no id". Ids are local to the
file:

- Before decoding, the reader takes a relocation base, the next id its
  process would mint. Every non-zero `HirId` decoded (map keys, fields and
  references alike) becomes `raw + base`, saturating at `0xFFFFFFFF`. `0`
  stays `0`.
- After decoding, the reader moves its id counter past the largest id in the
  module. For formats 0 to 2 it computes that id from the module id and the
  function, block, value, local, parameter, global, effect and handler ids.
  For format 3 it uses `base + max_id` from the Split directory.

A producer therefore needs only ids that are non-zero and unique within the
module. Small ids counted from 1 keep clear of the saturation.

**`LifetimeId`** is a u32. `0` is the static lifetime and `0xFFFFFFFF` an
anonymous one. Not relocated.

**`TypeId`** is a u32 type-registry id. Not relocated.

**`InternedString`** is encoded as its string. The reader interns it.

**`Span`** is `{ start: usize, end: usize, file: u32 }`: byte offsets into
source file number `file`. The source files themselves are not in the file.

### 5.2 Structures

The module:

```rust
HirModule {
    id: HirId,
    name: InternedString,
    functions: IndexMap<HirId, HirFunction>,
    globals: IndexMap<HirId, HirGlobal>,
    types: IndexMap<TypeId, HirType>,
    imports: Vec<HirImport>,
    exports: Vec<HirExport>,
    version: u64,                         // hot-reload generation
    dependencies: HashSet<HirId>,
    effects: IndexMap<HirId, HirEffect>,
    handlers: IndexMap<HirId, HirEffectHandler>,
    automatic_release: bool,
}
```

Functions and their bodies:

```rust
HirFunction {
    id: HirId,
    name: InternedString,
    signature: HirFunctionSignature,
    entry_block: HirId,
    blocks: IndexMap<HirId, HirBlock>,
    locals: IndexMap<HirId, HirLocal>,
    values: IndexMap<HirId, HirValue>,
    previous_version: Option<HirId>,
    is_external: bool,
    calling_convention: CallingConvention,
    attributes: FunctionAttributes,
    link_name: Option<String>,            // symbol name override
}

HirFunctionSignature {
    params: Vec<HirParam>,
    returns: Vec<HirType>,
    type_params: Vec<HirTypeParam>,
    const_params: Vec<HirConstParam>,
    lifetime_params: Vec<HirLifetime>,
    is_variadic: bool,
    is_async: bool,
    is_fiber: bool,
    effects: Vec<InternedString>,
    is_pure: bool,
}

HirParam { id: HirId, name: InternedString, ty: HirType,
           attributes: ParamAttributes, ownership: ParamOwnership }
ParamAttributes { by_ref: bool, sret: bool, zext: bool, sext: bool,
                  noalias: bool, nonnull: bool, readonly: bool }

FunctionAttributes {
    inline: InlineHint,
    no_return: bool, no_unwind: bool, no_inline: bool, always_inline: bool,
    cold: bool, hot: bool, pure: bool, const_fn: bool,
    strict_fp: bool, cooperative: bool, optimized: bool, deferred: bool,
    osr_region: bool,
    release_facts: Option<ReleaseFacts>,
}
ReleaseFacts { returns_owned: bool, returns_param: Vec<bool>, automatic_release: bool }

HirBlock {
    id: HirId,
    label: Option<InternedString>,
    phis: Vec<HirPhi>,
    instructions: Vec<HirInstruction>,
    terminator: HirTerminator,
    dominance_frontier: HashSet<HirId>,
    predecessors: Vec<HirId>,
    successors: Vec<HirId>,
}
HirPhi { result: HirId, ty: HirType, incoming: Vec<(HirId, HirId)> }   // (value, block)

HirValue { id: HirId, ty: HirType, kind: HirValueKind,
           uses: HashSet<HirId>, span: Option<Span> }
HirLocal { id: HirId, name: InternedString, ty: HirType,
           is_mutable: bool, lifetime: Option<HirLifetime> }
```

Generics, lifetimes and types:

```rust
HirLifetime { id: LifetimeId, name: Option<InternedString>, bounds: Vec<LifetimeBound> }
HirTypeParam { name: InternedString, constraints: Vec<TypeConstraint> }
HirConstParam { name: InternedString, ty: HirType, default: Option<HirConstant> }
HirMethodSignature { name: InternedString, params: Vec<HirType>, return_type: HirType,
                     is_static: bool, is_async: bool }

HirStructType { name: Option<InternedString>, fields: Vec<HirType>, packed: bool }
HirUnionType { name: Option<InternedString>, variants: Vec<HirUnionVariant>,
               discriminant_type: HirType, is_c_union: bool }
HirUnionVariant { name: InternedString, ty: HirType, discriminant: u64 }
HirFunctionType { params: Vec<HirType>, returns: Vec<HirType>,
                  lifetime_params: Vec<HirLifetime>, is_variadic: bool }
HirClosureType { function_type: HirFunctionType, captures: Vec<HirCapture>,
                 call_mode: HirClosureCallMode }
HirCapture { name: InternedString, ty: HirType, mode: HirCaptureMode }

HirVTable { id: HirId, trait_id: TypeId, for_type: HirType, methods: Vec<HirVTableEntry> }
HirVTableEntry { method_name: InternedString, function_id: HirId, signature: HirMethodSignature }

HirPattern { kind: HirPatternKind, target: HirId, bindings: Vec<HirPatternBinding> }
HirPatternBinding { name: InternedString, value_id: HirId, ty: HirType }
```

Globals, imports and exports:

```rust
HirGlobal { id: HirId, name: InternedString, ty: HirType,
            initializer: Option<HirConstant>, is_const: bool,
            is_thread_local: bool, linkage: Linkage, visibility: Visibility }
HirImport { name: InternedString, kind: ImportKind, attributes: ImportAttributes }
ImportAttributes { dll_import: Option<String>, weak: bool }
HirExport { name: InternedString, internal_name: InternedString, kind: ExportKind }
```

Algebraic effects:

```rust
HirEffect { id: HirId, name: InternedString, type_params: Vec<HirTypeParam>,
            operations: Vec<HirEffectOp> }
HirEffectOp { id: HirId, name: InternedString, type_params: Vec<HirTypeParam>,
              params: Vec<HirParam>, return_type: HirType }
HirEffectHandler { id: HirId, name: InternedString, effect_id: HirId,
                   type_params: Vec<HirTypeParam>, state_fields: Vec<HirHandlerField>,
                   implementations: Vec<HirEffectHandlerImpl> }
HirHandlerField { name: InternedString, ty: HirType }
HirEffectHandlerImpl { op_name: InternedString, type_params: Vec<HirTypeParam>,
                       params: Vec<HirParam>, return_type: HirType, entry_block: HirId,
                       blocks: IndexMap<HirId, HirBlock>, is_resumable: bool, is_async: bool }
```

### 5.3 Enumerations

The tag is the variant's position in its declaration; in Postcard it is a
varint, in Bincode a u32, in JSON the variant name.

#### Instructions (`HirInstruction`)

| Tag | Instruction | Fields, in order |
|-----|-------------|------------------|
| 0 | `Binary` | `op: BinaryOp, result: HirId, ty: HirType, left: HirId, right: HirId` |
| 1 | `Unary` | `op: UnaryOp, result: HirId, ty: HirType, operand: HirId` |
| 2 | `Alloca` | `result: HirId, ty: HirType, count: Option<HirId>, align: u32` |
| 3 | `Load` | `result: HirId, ty: HirType, ptr: HirId, align: u32, volatile: bool` |
| 4 | `Store` | `value: HirId, ptr: HirId, align: u32, volatile: bool` |
| 5 | `GetElementPtr` | `result: HirId, ty: HirType, ptr: HirId, indices: Vec<HirId>` |
| 6 | `Call` | `result: Option<HirId>, callee: HirCallable, args: Vec<HirId>, type_args: Vec<HirType>, const_args: Vec<HirConstant>, is_tail: bool` |
| 7 | `IndirectCall` | `result: Option<HirId>, func_ptr: HirId, args: Vec<HirId>, return_ty: HirType` |
| 8 | `Cast` | `op: CastOp, result: HirId, ty: HirType, operand: HirId` |
| 9 | `Select` | `result: HirId, ty: HirType, condition: HirId, true_val: HirId, false_val: HirId` |
| 10 | `ExtractValue` | `result: HirId, ty: HirType, aggregate: HirId, indices: Vec<u32>` |
| 11 | `InsertValue` | `result: HirId, ty: HirType, aggregate: HirId, value: HirId, indices: Vec<u32>` |
| 12 | `Atomic` | `op: AtomicOp, result: HirId, ty: HirType, ptr: HirId, value: Option<HirId>, ordering: AtomicOrdering` |
| 13 | `Fence` | `ordering: AtomicOrdering` |
| 14 | `CreateUnion` | `result: HirId, union_ty: HirType, variant_index: u32, value: HirId` |
| 15 | `GetUnionDiscriminant` | `result: HirId, union_val: HirId` |
| 16 | `ExtractUnionValue` | `result: HirId, ty: HirType, union_val: HirId, variant_index: u32` |
| 17 | `CreateTraitObject` | `result: HirId, trait_id: TypeId, data_ptr: HirId, vtable_id: HirId` |
| 18 | `UpcastTraitObject` | `result: HirId, sub_trait_object: HirId, sub_trait_id: TypeId, super_trait_id: TypeId, super_vtable_id: HirId` |
| 19 | `TraitMethodCall` | `result: Option<HirId>, trait_object: HirId, method_index: usize, method_sig: HirMethodSignature, args: Vec<HirId>, return_ty: HirType` |
| 20 | `CreateClosure` | `result: HirId, closure_ty: HirType, function: HirId, captures: Vec<HirId>` |
| 21 | `CallClosure` | `result: Option<HirId>, closure: HirId, args: Vec<HirId>` |
| 22 | `CreateRef` | `result: HirId, value: HirId, lifetime: HirLifetime, mutable: bool` |
| 23 | `Deref` | `result: HirId, ty: HirType, reference: HirId` |
| 24 | `Move` | `result: HirId, ty: HirType, source: HirId` |
| 25 | `Copy` | `result: HirId, ty: HirType, source: HirId` |
| 26 | `BeginLifetime` | `lifetime: HirLifetime` |
| 27 | `EndLifetime` | `lifetime: HirLifetime` |
| 28 | `LifetimeConstraint` | `longer: HirLifetime, shorter: HirLifetime` |
| 29 | `PerformEffect` | `result: Option<HirId>, effect_id: HirId, op_name: InternedString, args: Vec<HirId>, return_ty: HirType` |
| 30 | `HandleEffect` | `result: Option<HirId>, handler_id: HirId, handler_state: Vec<HirId>, body_block: HirId, continuation_block: HirId, return_ty: HirType` |
| 31 | `Resume` | `value: HirId, continuation: HirId` |
| 32 | `AbortEffect` | `value: HirId, handler_scope: HirId` |
| 33 | `CaptureContinuation` | `result: HirId, resume_ty: HirType` |
| 34 | `VectorSplat` | `result: HirId, ty: HirType, scalar: HirId` |
| 35 | `VectorExtractLane` | `result: HirId, ty: HirType, vector: HirId, lane: u8` |
| 36 | `VectorInsertLane` | `result: HirId, ty: HirType, vector: HirId, scalar: HirId, lane: u8` |
| 37 | `VectorHorizontalReduce` | `result: HirId, ty: HirType, vector: HirId, op: BinaryOp` |
| 38 | `VectorLoad` | `result: HirId, ty: HirType, ptr: HirId, align: u32` |
| 39 | `VectorStore` | `value: HirId, ptr: HirId, align: u32` |
| 40 | `VectorUnaryOp` | `result: HirId, ty: HirType, op: VectorUnaryKind, operand: HirId` |
| 41 | `VectorMinMax` | `result: HirId, ty: HirType, op: VectorMinMaxKind, left: HirId, right: HirId` |
| 42 | `VectorDot` | `result: HirId, acc: HirId, a: HirId, b: HirId, rhs_i7: bool, rhs_unsigned: bool` |
| 43 | `AsyncSaveSlot` | `frame: HirId, slot: u32, value: HirId` |
| 44 | `AsyncLoadSlot` | `result: HirId, ty: HirType, frame: HirId, slot: u32` |
| 45 | `FiberNew` | `result: HirId, ty: HirType, closure: HirId, stack_size: HirId` |
| 46 | `FiberResume` | `result: HirId, ty: HirType, fiber: HirId` |
| 47 | `FiberResumeWith` | `result: HirId, ty: HirType, fiber: HirId, value: HirId` |
| 48 | `FiberYield` | `value: HirId` |
| 49 | `FiberTransfer` | `result: HirId, ty: HirType, target: HirId, value: HirId` |
| 50 | `FiberCancel` | `fiber: HirId` |
| 51 | `FiberDrop` | `fiber: HirId` |

#### Terminators (`HirTerminator`)

| Tag | Terminator | Fields, in order |
|-----|------------|------------------|
| 0 | `Return` | `values: Vec<HirId>` |
| 1 | `Branch` | `target: HirId` |
| 2 | `CondBranch` | `condition: HirId, true_target: HirId, false_target: HirId` |
| 3 | `Switch` | `value: HirId, default: HirId, cases: Vec<(HirConstant, HirId)>` |
| 4 | `Unreachable` | none |
| 5 | `Invoke` | `callee: HirCallable, args: Vec<HirId>, normal: HirId, unwind: HirId` |
| 6 | `PatternMatch` | `value: HirId, patterns: Vec<HirPattern>, default: Option<HirId>` |

#### Types (`HirType`)

| Tag | Type | Fields, in order |
|-----|------|------------------|
| 0 to 15 | `Void`, `Bool`, `I8`, `I16`, `I32`, `I64`, `I128`, `U8`, `U16`, `U32`, `U64`, `U128`, `F32`, `F64`, `USize`, `ISize` | none |
| 16 | `Ptr` | `HirType` |
| 17 | `Ref` | `lifetime: HirLifetime, pointee: HirType, mutable: bool` |
| 18 | `Array` | `HirType, u64` (element, length) |
| 19 | `Vector` | `HirType, u32` (element, lanes) |
| 20 | `Struct` | `HirStructType` |
| 21 | `Union` | `HirUnionType` |
| 22 | `Function` | `HirFunctionType` |
| 23 | `Closure` | `HirClosureType` |
| 24 | `Opaque` | `InternedString` |
| 25 | `ConstGeneric` | `InternedString` |
| 26 | `Generic` | `base: HirType, type_args: Vec<HirType>, const_args: Vec<HirConstant>` |
| 27 | `TraitObject` | `trait_id: TypeId, vtable: Option<HirId>` |
| 28 | `Interface` | `methods: Vec<HirMethodSignature>, is_structural: bool` |
| 29 | `Promise` | `HirType` |
| 30 | `Fiber` | `HirType` |
| 31 | `AssociatedType` | `trait_id: TypeId, self_ty: HirType, name: InternedString` |
| 32 | `Continuation` | `resume_ty: HirType, result_ty: HirType` |
| 33 | `EffectRow` | `effects: Vec<InternedString>, tail: Option<InternedString>` |

`USize` and `ISize` are pointer-width integers, sized by the target.

#### Constants (`HirConstant`)

| Tag | Constant | Payload |
|-----|----------|---------|
| 0 | `Bool` | `bool` |
| 1 | `I8` | `i8` |
| 2 | `I16` | `i16` |
| 3 | `I32` | `i32` |
| 4 | `I64` | `i64` |
| 5 | `I128` | `i128` |
| 6 | `U8` | `u8` |
| 7 | `U16` | `u16` |
| 8 | `U32` | `u32` |
| 9 | `U64` | `u64` |
| 10 | `U128` | `u128` |
| 11 | `F32` | `f32` |
| 12 | `F64` | `f64` |
| 13 | `USize` | `u64` |
| 14 | `ISize` | `i64` |
| 15 | `Null` | `HirType` |
| 16 | `Array` | `Vec<HirConstant>` |
| 17 | `Struct` | `Vec<HirConstant>` |
| 18 | `String` | `InternedString` |
| 19 | `VTable` | `HirVTable` |

#### Values, callees, patterns, imports and exports

| Enum | Tag: variant (payload) |
|------|------------------------|
| `HirValueKind` | 0 `Parameter(u32)`, 1 `Instruction`, 2 `Constant(HirConstant)`, 3 `Global(HirId)`, 4 `Undef` |
| `HirCallable` | 0 `Function(HirId)`, 1 `Indirect(HirId)`, 2 `Intrinsic(Intrinsic)`, 3 `Symbol(String)`, 4 `FuncRef(HirId)` |
| `HirPatternKind` | 0 `Constant(HirConstant)`, 1 `UnionVariant { union_ty: HirType, variant_index: u32, inner_pattern: Option<HirPattern> }`, 2 `Struct { struct_ty: HirType, field_patterns: Vec<(u32, HirPattern)> }`, 3 `Wildcard`, 4 `Binding(InternedString)`, 5 `Guard { pattern: HirPattern, condition: HirId }` |
| `TypeConstraint` | 0 `Trait(InternedString)`, 1 `Subtype(HirType)`, 2 `Sized` |
| `LifetimeBound` | 0 `Outlives(LifetimeId)`, 1 `Static` |
| `ImportKind` | 0 `Function(HirFunctionSignature)`, 1 `Global(HirType)`, 2 `Type { ty: HirType, type_id: TypeId }` |
| `ExportKind` | 0 `Function(HirId)`, 1 `Global(HirId)` |

#### Operators and plain tags

Every variant below carries no payload.

| Enum | Tags |
|------|------|
| `BinaryOp` | 0 `Add`, 1 `Sub`, 2 `Mul`, 3 `Div`, 4 `Rem`, 5 `And`, 6 `Or`, 7 `Xor`, 8 `Shl`, 9 `Shr`, 10 `Eq`, 11 `Ne`, 12 `Lt`, 13 `Le`, 14 `Gt`, 15 `Ge`, 16 `FAdd`, 17 `FSub`, 18 `FMul`, 19 `FDiv`, 20 `FRem`, 21 `FEq`, 22 `FNe`, 23 `FLt`, 24 `FLe`, 25 `FGt`, 26 `FGe` |
| `UnaryOp` | 0 `Neg`, 1 `Not`, 2 `FNeg` |
| `CastOp` | 0 `Trunc`, 1 `ZExt`, 2 `SExt`, 3 `FpTrunc`, 4 `FpExt`, 5 `FpToUi`, 6 `FpToSi`, 7 `UiToFp`, 8 `SiToFp`, 9 `PtrToInt`, 10 `IntToPtr`, 11 `Bitcast` |
| `AtomicOp` | 0 `Load`, 1 `Store`, 2 `Exchange`, 3 `Add`, 4 `Sub`, 5 `And`, 6 `Or`, 7 `Xor`, 8 `CompareExchange` |
| `AtomicOrdering` | 0 `Relaxed`, 1 `Acquire`, 2 `Release`, 3 `AcqRel`, 4 `SeqCst` |
| `VectorUnaryKind` | 0 `Sqrt`, 1 `Abs`, 2 `Neg`, 3 `Ceil`, 4 `Floor`, 5 `Trunc`, 6 `Round` |
| `VectorMinMaxKind` | 0 `Min`, 1 `Max` |
| `Intrinsic` | 0 `Memcpy`, 1 `Memset`, 2 `Memmove`, 3 `Sqrt`, 4 `Rsqrt`, 5 `Fabs`, 6 `Floor`, 7 `Fma`, 8 `Sin`, 9 `Cos`, 10 `Pow`, 11 `Log`, 12 `Exp`, 13 `Ctpop`, 14 `Ctlz`, 15 `Cttz`, 16 `Bswap`, 17 `SizeOf`, 18 `AlignOf`, 19 `AddWithOverflow`, 20 `SubWithOverflow`, 21 `MulWithOverflow`, 22 `Malloc`, 23 `Free`, 24 `Realloc`, 25 `Drop`, 26 `IncRef`, 27 `DecRef`, 28 `Alloca`, 29 `GCSafepoint`, 30 `Await`, 31 `Yield`, 32 `Panic`, 33 `Abort`, 34 `ClosureToZrtl`, 35 `BoxToZrtl`, 36 `PrimitiveToBox`, 37 `TypeTagOf` |
| `CallingConvention` | 0 `Fast`, 1 `C`, 2 `System`, 3 `WebKit` |
| `InlineHint` | 0 `None`, 1 `Hint`, 2 `Always`, 3 `Never` |
| `Linkage` | 0 `External`, 1 `Internal`, 2 `Private`, 3 `Weak`, 4 `LinkOnce` |
| `Visibility` | 0 `Default`, 1 `Hidden`, 2 `Protected` |
| `HirCaptureMode` | 0 `ByValue`, 1 `ByRef`, 2 `ByMutRef` |
| `HirClosureCallMode` | 0 `Once`, 1 `Fn`, 2 `FnMut` |
| `ParamOwnership` | 0 `Copied`, 1 `Borrowed`, 2 `BorrowedMut`, 3 `Owned`, 4 `Shared` |

### 5.4 Module invariants

The reader decodes whatever parses. The compiler passes that consume the
module rely on the following, as `HirBuilder` and the SSA lowering produce
them:

- Every map key equals the `id` of its entry (`functions`, `globals`,
  `effects`, `handlers`, `blocks`, `locals`, `values`).
- `entry_block` is a key of `blocks`. A function with `is_external = true`
  has no blocks, locals or values.
- Parameter `i` of a function has a `values` entry under its `HirParam.id`
  with kind `Parameter(i)`.
- Every value id an instruction, phi or terminator defines or uses is a key
  of `values`; a constant operand is a value of kind `Constant`.
- Every block id a terminator, phi or `HandleEffect` names is a key of
  `blocks`.
- `predecessors`, `successors`, `dominance_frontier` and `uses` are analysis
  results that some passes read as written; a producer fills them to match
  the code.

---

## 6. Split Images (format 3)

A Split image encodes each function's shell and body on its own, so a reader
can open the image by decoding only its directory, then decode one function
when it is asked for.

### 6.1 Layout

The payload is a Postcard `SplitPayload` followed immediately by a blob of
`blob_len` bytes:

```rust
SplitPayload {
    stripped: Extent,        // the module, with an empty `functions` map
    directory: Vec<FnEntry>, // one entry per function, in module order
    by_id: Vec<u32>,         // directory positions, ascending by FnEntry.id
    by_name: Vec<u32>,       // directory positions, ascending by name bytes; ties in directory order
    blob_len: u32,           // length of the blob
    max_id: u32,             // largest HirId anywhere in the module, as written
}

FnEntry {
    id: HirId,
    name: Extent,            // the function's name, raw UTF-8, no length prefix
    shell: Extent,           // Postcard HirFunction with empty blocks, locals and values;
                             // an external function whole
    body: Option<Extent>,    // Postcard HirFunction, whole; None for an external function
}

Extent { at: u32, len: u32 } // a byte range of the blob, `at` counted from the blob's start
```

Each extent addresses a separately encoded Postcard value (or, for `name`,
raw bytes). The writer lays the blob out as the stripped module, then for
each function its name, shell and body; readers go only by the extents. A
shell and a body carry the same id, name, signature and attributes.

### 6.2 What a reader checks

On open, beyond section 2.1:

- The bytes after `SplitPayload` number exactly `blob_len`, else
  `InvalidFormat`.
- `by_id` and `by_name` each have one entry per directory entry, else
  `InvalidFormat`.

Extents, positions and sort order are not checked on open. An extent that
falls outside the blob, or a shell or body that does not decode, leaves that
function unavailable. The stripped module must decode.

### 6.3 Decoding

All parts of one image (the directory, the stripped module, every shell and
every body) are relocated by the same base, taken when the image is opened.
The reader then moves its id counter past `base + max_id` without scanning
the module, so `max_id` must be at least every id in it.

Lookups:

- By id: binary search of `by_id`, comparing directory ids.
- By name: binary search of `by_name`, comparing name bytes; the first match
  in module order wins.

`deserialize_module` on a Split image returns the whole module: the stripped
module plus, for each directory entry in order, its body (or its shell when
it has no body), keyed by the entry's id.

---

## 7. Writing a File

1. Build the `HirModule` (section 5), with non-zero ids unique within the
   module.
2. Encode it with the chosen codec (section 3). For Split, lay out the blob
   and directory as in section 6.1.
3. Compute the CRC-32 of the encoded payload.
4. Write the 44-byte header (section 2) with version 3.0, the format byte,
   flags 0, the module id, the payload length and the checksum; then the
   payload; then nothing else.

The Rust entry points are `serialize_module(&module, Format)`,
`serialize_module_to_writer` and `serialize_module_to_file`, and
`deserialize_module`, `deserialize_module_from_reader`,
`deserialize_module_from_file`, `deserialize_module_lazy`,
`deserialize_module_lazy_trusted` and `bytecode_stats` for reading.

---

## 8. Versioning

A reader accepts a file only when its major version equals the reader's own,
3. The minor version is written as 0 and not compared. A file of any other
major version is refused with `VersionMismatch`; there is no reading of older
or newer majors. The HIR caches in this repository treat any read error,
`VersionMismatch` included, as a miss and recompile.

The major version moves when the bytecode's own layout changes: the header,
the Split structures, or the encoding of identifiers. It does not track the
HIR schema of section 5. Adding, removing or reordering a field or a variant
of any type there also changes the payload layout, since Postcard and Bincode
are positional and tags are declaration positions, so a payload must match
the declarations of the build that reads it. Language snapshots pair each
Split image with the compiler's build id and target pointer width for this
reason, and do not read an image recorded by a different build or for a
different width.

### Version history

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2025-11-16 | Header plus a Postcard, JSON or Bincode encoding of `HirModule`; `HirId` and `LifetimeId` encoded as UUIDs. |
| 1.0 | 2025-11-27 | Header written as the fixed 44-byte little-endian record instead of a Bincode-encoded struct; version unchanged. |
| 2.0 | 2026-06-05 | `HirId` and `LifetimeId` become u32; the header's `module_id` carries the u32 id zero-extended to 16 bytes. |
| 2.0 | 2026-09-14 | Format 3 (Split) added: a stripped module with each function body encoded separately; version unchanged. |
| 3.0 | 2026-09-24 | Split payload becomes a function directory with per-function name, shell and body extents into one blob, plus `by_id`, `by_name` and `blob_len`. |
